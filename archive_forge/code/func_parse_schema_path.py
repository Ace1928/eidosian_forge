from collections import OrderedDict
import functools
import os
import re
from typing import (
from google.pubsub_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.api_core import timeout as timeouts  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from google.pubsub_v1.services.publisher import pagers
from google.pubsub_v1.types import pubsub
from google.pubsub_v1.types import TimeoutType
import grpc
from .transports.base import PublisherTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import PublisherGrpcTransport
from .transports.grpc_asyncio import PublisherGrpcAsyncIOTransport
from .transports.rest import PublisherRestTransport
@staticmethod
def parse_schema_path(path: str) -> Dict[str, str]:
    """Parses a schema path into its component segments."""
    m = re.match('^projects/(?P<project>.+?)/schemas/(?P<schema>.+?)$', path)
    return m.groupdict() if m else {}