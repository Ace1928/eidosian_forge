import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import gapic_v1, path_template, rest_helpers, rest_streaming
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechTransport
@property
def synthesize_speech(self) -> Callable[[cloud_tts.SynthesizeSpeechRequest], cloud_tts.SynthesizeSpeechResponse]:
    return self._SynthesizeSpeech(self._session, self._host, self._interceptor)