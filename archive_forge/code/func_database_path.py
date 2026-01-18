from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Iterable, Sequence, Tuple, Type, Union, cast
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from cloudsdk.google.protobuf import struct_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.services.spanner import pagers
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import commit_response
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import mutation
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction
from .transports.base import SpannerTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import SpannerGrpcTransport
from .transports.grpc_asyncio import SpannerGrpcAsyncIOTransport
from .transports.rest import SpannerRestTransport
@staticmethod
def database_path(project: str, instance: str, database: str) -> str:
    """Returns a fully-qualified database string."""
    return 'projects/{project}/instances/{instance}/databases/{database}'.format(project=project, instance=instance, database=database)