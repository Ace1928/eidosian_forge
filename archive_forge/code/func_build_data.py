from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
def build_data(self, audio_data: AudioData) -> bytes:
    flac_data = audio_data.get_flac_data(convert_rate=self.to_convert_rate(audio_data.sample_rate), convert_width=2)
    return flac_data