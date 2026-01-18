from collections import OrderedDict
import os
import re
from typing import (
import warnings
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.cloud.speech_v1p1beta1 import gapic_version as package_version
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from google.cloud.speech_v1p1beta1.types import cloud_speech
from .transports.base import DEFAULT_CLIENT_INFO, SpeechTransport
from .transports.grpc import SpeechGrpcTransport
from .transports.grpc_asyncio import SpeechGrpcAsyncIOTransport
from .transports.rest import SpeechRestTransport
@staticmethod
def parse_custom_class_path(path: str) -> Dict[str, str]:
    """Parses a custom_class path into its component segments."""
    m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/customClasses/(?P<custom_class>.+?)$', path)
    return m.groupdict() if m else {}