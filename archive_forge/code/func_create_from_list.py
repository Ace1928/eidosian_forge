from __future__ import annotations
from typing import (
import param
from ..models.speech_to_text import SpeechToText as _BkSpeechToText
from .base import Widget
from .button import BUTTON_TYPES
@classmethod
def create_from_list(cls, results):
    """
        Deserializes a list of serialized RecognitionResults.
        """
    return [cls.create_from_dict(result) for result in results]