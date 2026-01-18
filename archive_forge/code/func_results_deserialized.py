from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
@property
def results_deserialized(self):
    """
        Returns the results as a List of RecognitionResults
        """
    return RecognitionResult.create_from_list(self.results)