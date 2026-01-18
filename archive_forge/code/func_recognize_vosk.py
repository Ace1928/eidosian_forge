from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
def recognize_vosk(self, audio_data, language='en'):
    from vosk import KaldiRecognizer, Model
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    if not hasattr(self, 'vosk_model'):
        if not os.path.exists('model'):
            return "Please download the model from https://github.com/alphacep/vosk-api/blob/master/doc/models.md and unpack as 'model' in the current folder."
            exit(1)
        self.vosk_model = Model('model')
    rec = KaldiRecognizer(self.vosk_model, 16000)
    rec.AcceptWaveform(audio_data.get_raw_data(convert_rate=16000, convert_width=2))
    finalRecognition = rec.FinalResult()
    return finalRecognition