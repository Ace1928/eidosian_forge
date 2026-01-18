from typing import Any, Dict, List
import pytest
import panel as pn
from panel.widgets import TextToSpeech, Utterance, Voice
def test_can_speak(document, comm):
    text = 'Give me back my money!'
    speaker = TextToSpeech()
    speaker.value = text
    model = speaker.get_root(document, comm)
    assert model.speak['text'] == text