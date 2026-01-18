from typing import Any, Dict, List
import pytest
import panel as pn
from panel.widgets import TextToSpeech, Utterance, Voice
def test_can_set_voices():
    voices = Voice.to_voices_list(_VOICES_CHROME_WIN10)
    utterance = Utterance()
    utterance.set_voices(voices)
    assert utterance.param.lang.default == 'en-US'
    assert utterance.lang == 'en-US'
    assert utterance.param.voice.default.lang == 'en-US'
    assert utterance.voice == utterance.param.voice.default