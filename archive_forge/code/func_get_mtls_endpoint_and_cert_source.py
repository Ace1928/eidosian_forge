from collections import OrderedDict
import os
import re
from typing import Dict, Mapping, MutableMapping, MutableSequence, Optional, Iterable, Iterator, Sequence, Tuple, Type, Union, cast
import warnings
from googlecloudsdk.generated_clients.gapic_clients.storage_v2 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials             # type: ignore
from google.auth.transport import mtls                            # type: ignore
from google.auth.transport.grpc import SslCredentials             # type: ignore
from google.auth.exceptions import MutualTLSChannelError          # type: ignore
from google.oauth2 import service_account                         # type: ignore
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.storage_v2.services.storage import pagers
from googlecloudsdk.generated_clients.gapic_clients.storage_v2.types import storage
from .transports.base import StorageTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import StorageGrpcTransport
from .transports.grpc_asyncio import StorageGrpcAsyncIOTransport
from .transports.rest import StorageRestTransport
@classmethod
def get_mtls_endpoint_and_cert_source(cls, client_options: Optional[client_options_lib.ClientOptions]=None):
    """Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """
    if client_options is None:
        client_options = client_options_lib.ClientOptions()
    use_client_cert = os.getenv('GOOGLE_API_USE_CLIENT_CERTIFICATE', 'false')
    use_mtls_endpoint = os.getenv('GOOGLE_API_USE_MTLS_ENDPOINT', 'auto')
    if use_client_cert not in ('true', 'false'):
        raise ValueError('Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`')
    if use_mtls_endpoint not in ('auto', 'never', 'always'):
        raise MutualTLSChannelError('Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`')
    client_cert_source = None
    if use_client_cert == 'true':
        if client_options.client_cert_source:
            client_cert_source = client_options.client_cert_source
        elif mtls.has_default_client_cert_source():
            client_cert_source = mtls.default_client_cert_source()
    if client_options.api_endpoint is not None:
        api_endpoint = client_options.api_endpoint
    elif use_mtls_endpoint == 'always' or (use_mtls_endpoint == 'auto' and client_cert_source):
        api_endpoint = cls.DEFAULT_MTLS_ENDPOINT
    else:
        api_endpoint = cls.DEFAULT_ENDPOINT
    return (api_endpoint, client_cert_source)