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
def ssl_channel_credentials(root_certificates=None, private_key=None, certificate_chain=None):
    """Creates a ChannelCredentials for use with an SSL-enabled Channel.

    Args:
      root_certificates: The PEM-encoded root certificates as a byte string,
        or None to retrieve them from a default location chosen by gRPC
        runtime.
      private_key: The PEM-encoded private key as a byte string, or None if no
        private key should be used.
      certificate_chain: The PEM-encoded certificate chain as a byte string
        to use or None if no certificate chain should be used.

    Returns:
      A ChannelCredentials for use with an SSL-enabled Channel.
    """
    return ChannelCredentials(_cygrpc.SSLChannelCredentials(root_certificates, private_key, certificate_chain))