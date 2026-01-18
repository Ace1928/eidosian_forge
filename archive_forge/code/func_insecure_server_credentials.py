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
def insecure_server_credentials():
    """Creates a credentials object directing the server to use no credentials.
      This is an EXPERIMENTAL API.

    This object cannot be used directly in a call to `add_secure_port`.
    Instead, it should be used to construct other credentials objects, e.g.
    with xds_server_credentials.
    """
    return ServerCredentials(_cygrpc.insecure_server_credentials())