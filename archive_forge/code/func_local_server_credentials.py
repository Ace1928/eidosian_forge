import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def local_server_credentials(local_connect_type=LocalConnectionType.LOCAL_TCP):
    """Creates a local ServerCredentials used for local connections.

    This is an EXPERIMENTAL API.

    Local credentials are used by local TCP endpoints (e.g. localhost:10000)
    also UDS connections.

    The connections created by local server credentials are not
    encrypted, but will be checked if they are local or not.
    The UDS connections are considered secure by providing peer authentication
    and data confidentiality while TCP connections are considered insecure.

    It is allowed to transmit call credentials over connections created by local
    server credentials.

    Local server credentials are useful for 1) eliminating insecure_channel usage;
    2) enable unit testing for call credentials without setting up secrets.

    Args:
      local_connect_type: Local connection type (either
        grpc.LocalConnectionType.UDS or grpc.LocalConnectionType.LOCAL_TCP)

    Returns:
      A ServerCredentials for use with a local Server
    """
    return ServerCredentials(_cygrpc.server_credentials_local(local_connect_type.value))