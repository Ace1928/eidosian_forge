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
from google.cloud.speech_v2 import gapic_version as package_version
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.location import locations_pb2  # type: ignore
from google.longrunning import operations_pb2  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.cloud.speech_v2.services.speech import pagers
from google.cloud.speech_v2.types import cloud_speech
from .transports.base import DEFAULT_CLIENT_INFO, SpeechTransport
from .transports.grpc import SpeechGrpcTransport
from .transports.grpc_asyncio import SpeechGrpcAsyncIOTransport
from .transports.rest import SpeechRestTransport
@staticmethod
def crypto_key_version_path(project: str, location: str, key_ring: str, crypto_key: str, crypto_key_version: str) -> str:
    """Returns a fully-qualified crypto_key_version string."""
    return 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key, crypto_key_version=crypto_key_version)