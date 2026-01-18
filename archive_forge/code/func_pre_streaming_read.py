from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import commit_response
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction
from .base import SpannerTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
def pre_streaming_read(self, request: spanner.ReadRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ReadRequest, Sequence[Tuple[str, str]]]:
    """Pre-rpc interceptor for streaming_read

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
    return (request, metadata)