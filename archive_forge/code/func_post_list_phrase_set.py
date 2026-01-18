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
from google.protobuf import empty_pb2  # type: ignore
from google.cloud.speech_v1p1beta1.types import cloud_speech_adaptation, resource
from .base import AdaptationTransport
from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
def post_list_phrase_set(self, response: cloud_speech_adaptation.ListPhraseSetResponse) -> cloud_speech_adaptation.ListPhraseSetResponse:
    """Post-rpc interceptor for list_phrase_set

        Override in a subclass to manipulate the response
        after it is returned by the Adaptation server but before
        it is returned to user code.
        """
    return response