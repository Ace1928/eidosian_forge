from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
def create_request_builder(*, key: str | None=None, language: str='en-US', filter_level: ProfanityFilterLevel=0) -> RequestBuilder:
    if not isinstance(language, str):
        raise ValueError('``language`` must be a string')
    if key is not None and (not isinstance(key, str)):
        raise ValueError('``key`` must be ``None`` or a string')
    if key is None:
        key = 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'
    return RequestBuilder(key=key, language=language, filter_level=filter_level)