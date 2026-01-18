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
def recognize_ibm(self, audio_data, key, language='en-US', show_all=False):
    """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using the IBM Speech to Text API.

        The IBM Speech to Text username and password are specified by ``username`` and ``password``, respectively. Unfortunately, these are not available without `signing up for an account <https://console.ng.bluemix.net/registration/>`__. Once logged into the Bluemix console, follow the instructions for `creating an IBM Watson service instance <https://www.ibm.com/watson/developercloud/doc/getting_started/gs-credentials.shtml>`__, where the Watson service is "Speech To Text". IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX, while passwords are mixed-case alphanumeric strings.

        The recognition language is determined by ``language``, an RFC5646 language tag with a dialect like ``"en-US"`` (US English) or ``"zh-CN"`` (Mandarin Chinese), defaulting to US English. The supported language values are listed under the ``model`` parameter of the `audio recognition API documentation <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__, in the form ``LANGUAGE_BroadbandModel``, where ``LANGUAGE`` is the language value.

        Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the `raw API response <https://www.ibm.com/watson/developercloud/speech-to-text/api/v1/#sessionless_methods>`__ as a JSON dictionary.

        Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if the speech recognition operation failed, if the key isn't valid, or if there is no internet connection.
        """
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    assert isinstance(key, str), '``key`` must be a string'
    flac_data = audio_data.get_flac_data(convert_rate=None if audio_data.sample_rate >= 16000 else 16000, convert_width=None if audio_data.sample_width >= 2 else 2)
    url = 'https://gateway-wdc.watsonplatform.net/speech-to-text/api/v1/recognize'
    request = Request(url, data=flac_data, headers={'Content-Type': 'audio/x-flac'})
    request.get_method = lambda: 'POST'
    username = 'apikey'
    password = key
    authorization_value = base64.standard_b64encode('{}:{}'.format(username, password).encode('utf-8')).decode('utf-8')
    request.add_header('Authorization', 'Basic {}'.format(authorization_value))
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
    if 'results' not in result or len(result['results']) < 1 or 'alternatives' not in result['results'][0]:
        raise UnknownValueError()
    transcription = []
    confidence = None
    for utterance in result['results']:
        if 'alternatives' not in utterance:
            raise UnknownValueError()
        for hypothesis in utterance['alternatives']:
            if 'transcript' in hypothesis:
                transcription.append(hypothesis['transcript'])
                confidence = hypothesis['confidence']
                break
    return ('\n'.join(transcription), confidence)