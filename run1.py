# -----------------------------------------------------------
# 🧠 UNSTRUCTURED DATA ANALYZER — FINAL STREAMLIT APP
# -----------------------------------------------------------

import streamlit as st
#from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
import io
import random
import numpy as np
from PIL import Image
from rembg import remove
from deepface import DeepFace
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
import nltk

# -----------------------------------------------------------
# Setup
# -----------------------------------------------------------
st.set_page_config(page_title="🧠 Unstructured Data Analyzer", layout="wide")
st.title("🧠 Unstructured Data Analysis")

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model (cached)
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🖼 Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ===========================================================
# 🖼 TAB 1: IMAGE ANALYSIS
# ===========================================================
with tab1:
    st.header("🖼 Image Analysis with DeepFace")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        img_array = np.array(img)

        col1, col2, col3, col4 = st.columns(4)

        # Face Detection
        with col1:
            if st.button("Detect Face"):
                try:
                    detection = DeepFace.detectFace(img_array, enforce_detection=True)
                    st.success("✅ Face detected!")
                    st.image(detection, caption="Detected Face", use_column_width=True)
                except Exception as e:
                    st.error(f"Face detection failed: {e}")

        # Age & Gender
        with col2:
            if st.button("Detect Age & Gender"):
                try:
                    analysis = DeepFace.analyze(img_array, actions=['age', 'gender'], enforce_detection=True)
                    predicted_age = analysis[0]['age']
                    predicted_gender = analysis[0]['dominant_gender']
                    st.success("✅ Age & Gender detected!")
                    st.write(f"*Predicted Age:* {predicted_age}")
                    st.write(f"*Predicted Gender:* {predicted_gender}")
                except Exception as e:
                    st.error(f"Age/Gender detection failed: {e}")

        # Emotion Detection
        with col3:
            if st.button("Detect Emotion"):
                try:
                    analysis = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
                    predicted_emotion = analysis[0]['dominant_emotion']
                    st.success("✅ Emotion detected!")
                    st.write(f"*Predicted Emotion:* {predicted_emotion}")
                except Exception as e:
                    st.error(f"Emotion detection failed: {e}")

        # Background Removal
        with col4:
            if st.button("Remove Background"):
                try:
                    output_image = remove(img)
                    st.success("✅ Background removed!")
                    st.image(output_image, caption="BG Removed Image", width=300)
                except Exception as e:
                    st.error(f"Background removal failed: {e}")

# ===========================================================
# 🎧 TAB 2: AUDIO ANALYSIS
# ===========================================================
with tab2:
    # -------------------------------------------------------
    # 🗣 TEXT TO SPEECH
    # -------------------------------------------------------
    st.header("🗣 Text to Speech")

    text = st.text_area("Enter text to convert to speech:")

    if st.button("Convert to Audio"):
        if text.strip():
            tts = gTTS(text, lang='en')
            tts.save("output.mp3")
            with open("output.mp3", "rb") as audio_file:
                st.audio(audio_file.read(), format='audio/mp3')
            st.success("✅ Conversion complete!")
        else:
            st.warning("Please enter some text.")

    st.markdown("---")

    # -------------------------------------------------------
    # 🎙 SPEECH TO TEXT (Corrected & Error-Free)
    # -------------------------------------------------------
    st.header("🎙 Speech to Text")

    uploaded_audio = st.file_uploader("Upload audio file (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])

    if uploaded_audio is not None:
        try:
            # Read uploaded file bytes
            audio_bytes = uploaded_audio.read()

            # Convert audio to WAV format
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            # Play uploaded audio
            st.audio(wav_io, format="audio/wav")

            recognizer = sr.Recognizer()

            if st.button("📝 Transcribe Audio"):
                with st.spinner("Transcribing... please wait..."):
                    with sr.AudioFile(wav_io) as source:
                        audio_data = recognizer.record(source)

                    try:
                        text_output = recognizer.recognize_google(audio_data)
                        st.success("✅ Transcription complete!")
                        st.subheader("🗒 Transcribed Text:")
                        st.write(text_output)
                    except sr.UnknownValueError:
                        st.error("❌ Speech not recognized. Try clearer audio.")
                    except sr.RequestError:
                        st.error("⚠ Google Speech API unavailable or network issue.")
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    else:
        st.info("Please upload an audio file to begin transcription.")

# ===========================================================
# 📝 TAB 3: TEXT ANALYSIS
# ===========================================================
with tab3:
    st.header("📝 Text Analysis")

    # Sample stories
    stories = [
        "In a distant kingdom, Princess Elara explored magical forests and learned ancient secrets...",
        "Detective Samuel Hart roamed the streets of 1920s New York, solving impossible crimes...",
        "Captain Rhea explored an alien planet, uncovering ruins of an ancient civilization...",
        "Akira, a young coder in Tokyo, developed an AI to revolutionize traffic systems...",
        "Deep in the Amazon rainforest, scientists searched for plants with healing powers..."
    ]

    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    st.session_state.text_area = st.text_area(
        "Paste or modify your text here:",
        value=st.session_state.text_area,
        height=250
    )

    # Text Analysis
    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()

        if text:
            blob = TextBlob(text)
            words_and_tags = blob.tags
            nouns = [w for w, t in words_and_tags if t.startswith('NN')]
            verbs = [w for w, t in words_and_tags if t.startswith('VB')]
            adjectives = [w for w, t in words_and_tags if t.startswith('JJ')]
            adverbs = [w for w, t in words_and_tags if t.startswith('RB')]

            def make_wordcloud(words, color):
                if not words:
                    return None
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(" ".join(words))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)
            with col2:
                st.markdown("### ⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)
            with col3:
                st.markdown("### 🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)
            with col4:
                st.markdown("### 💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            st.markdown("### 📊 POS Counts")
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })
        else:
            st.warning("Please enter or select text first.")

    st.markdown("---")
    st.subheader("🧩 Named Entity Recognition (NER)")

    text = st.session_state.get('text_area', '')

    if text.strip():
        if st.button("🔍 Run NER Analysis"):
            doc = nlp(text)
            html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown("*Detected Entities:*", unsafe_allow_html=True)
            st.markdown(html, unsafe_allow_html=True)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.markdown("*Entity Table:*")
                st.table(entities)
            else:
                st.info("No named entities found.")
    else:

        st.info("Paste or select some text to see NER results.")
