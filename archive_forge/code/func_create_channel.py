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
@classmethod
def create_channel(cls, host: str='storage.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, quota_project_id: Optional[str]=None, **kwargs) -> aio.Channel:
    """Create and return a gRPC AsyncIO channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            aio.Channel: A gRPC AsyncIO channel object.
        """
    return grpc_helpers_async.create_channel(host, credentials=credentials, credentials_file=credentials_file, quota_project_id=quota_project_id, default_scopes=cls.AUTH_SCOPES, scopes=scopes, default_host=cls.DEFAULT_HOST, **kwargs)