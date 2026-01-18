import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
def test_can_deserialize_alternative():
    alternative = {'confidence': 0.9190853834152222, 'transcript': 'but why'}
    actual = RecognitionAlternative(**alternative)
    assert actual.confidence == alternative['confidence']
    assert actual.transcript == alternative['transcript']