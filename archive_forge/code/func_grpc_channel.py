import warnings
from typing import Awaitable, Callable, Dict, Optional, Sequence, Tuple, Union
from google.api_core import gapic_v1
from google.api_core import grpc_helpers_async
from google.auth import credentials as ga_credentials   # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
import grpc                        # type: ignore
from grpc.experimental import aio  # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.storage_v2.types import storage
from .base import StorageTransport, DEFAULT_CLIENT_INFO
from .grpc import StorageGrpcTransport
@property
def grpc_channel(self) -> aio.Channel:
    """Create the channel designed to connect to this service.

        This property caches on the instance; repeated calls return
        the same channel.
        """
    return self._grpc_channel