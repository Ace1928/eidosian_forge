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
from google.api_core import operations_v1
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_config
from google.longrunning import operations_pb2  # type: ignore
from .base import ConfigServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
@property
def operations_client(self) -> operations_v1.AbstractOperationsClient:
    """Create the client designed to process long-running operations.

        This property caches on the instance; repeated calls return the same
        client.
        """
    if self._operations_client is None:
        http_options: Dict[str, List[Dict[str, str]]] = {}
        rest_transport = operations_v1.OperationsRestTransport(host=self._host, credentials=self._credentials, scopes=self._scopes, http_options=http_options, path_prefix='v2')
        self._operations_client = operations_v1.AbstractOperationsClient(transport=rest_transport)
    return self._operations_client