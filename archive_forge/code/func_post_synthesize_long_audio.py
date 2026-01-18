import dataclasses
import json  # type: ignore
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.api_core import (
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts_lrs
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import TextToSpeechLongAudioSynthesizeTransport
def post_synthesize_long_audio(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
    """Post-rpc interceptor for synthesize_long_audio

        Override in a subclass to manipulate the response
        after it is returned by the TextToSpeechLongAudioSynthesize server but before
        it is returned to user code.
        """
    return response