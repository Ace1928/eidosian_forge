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
from google.cloud.location import locations_pb2  # type: ignore
from google.protobuf import json_format
import grpc  # type: ignore
from requests import __version__ as requests_version
from google.longrunning import operations_pb2  # type: ignore
from google.cloud.speech_v2.types import cloud_speech
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .base import SpeechTransport
def post_list_recognizers(self, response: cloud_speech.ListRecognizersResponse) -> cloud_speech.ListRecognizersResponse:
    """Post-rpc interceptor for list_recognizers

        Override in a subclass to manipulate the response
        after it is returned by the Speech server but before
        it is returned to user code.
        """
    return response