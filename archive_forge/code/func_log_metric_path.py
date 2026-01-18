from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Sequence, Tuple, Type, Union, cast
from googlecloudsdk.generated_clients.gapic_clients.logging_v2 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from google.api import distribution_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.services.metrics_service_v2 import pagers
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_metrics
from .transports.base import MetricsServiceV2Transport, DEFAULT_CLIENT_INFO
from .transports.grpc import MetricsServiceV2GrpcTransport
from .transports.grpc_asyncio import MetricsServiceV2GrpcAsyncIOTransport
from .transports.rest import MetricsServiceV2RestTransport
@staticmethod
def log_metric_path(project: str, metric: str) -> str:
    """Returns a fully-qualified log_metric string."""
    return 'projects/{project}/metrics/{metric}'.format(project=project, metric=metric)