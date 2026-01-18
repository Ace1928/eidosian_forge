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
def recognize_wit(self, audio_data, key, show_all=False):
    """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the Wit.ai API.

        The Wit.ai API key is specified by ``key``. Unfortunately, these are not available without `signing up for an account <https://wit.ai/>`__ and creating an app. You will need to add at least one intent to the app before you can see the API key, though the actual intent settings don't matter.

        To get the API key for a Wit.ai app, go to the app's overview page, go to the section titled "Make an API request", and look for something along the lines of ``Authorization: Bearer XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX``; ``XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`` is the API key. Wit.ai API keys are 32-character uppercase alphanumeric strings.

        The recognition language is configured in the Wit.ai app settings.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://wit.ai/docs/http/20141022#get-intent-via-text-link>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    assert isinstance(key, str), '``key`` must be a string'
    wav_data = audio_data.get_wav_data(convert_rate=None if audio_data.sample_rate >= 8000 else 8000, convert_width=2)
    url = 'https://api.wit.ai/speech?v=20170307'
    request = Request(url, data=wav_data, headers={'Authorization': 'Bearer {}'.format(key), 'Content-Type': 'audio/wav'})
    try:
        response = urlopen(request, timeout=self.operation_timeout)
    except HTTPError as e:
        raise RequestError('recognition request failed: {}'.format(e.reason))
    except URLError as e:
        raise RequestError('recognition connection failed: {}'.format(e.reason))
    response_text = response.read().decode('utf-8')
    result = json.loads(response_text)
    if show_all:
        return result
    if '_text' not in result or result['_text'] is None:
        raise UnknownValueError()
    return result['_text']