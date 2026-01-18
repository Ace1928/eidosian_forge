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
def ssl_server_credentials(private_key_certificate_chain_pairs, root_certificates=None, require_client_auth=False):
    """Creates a ServerCredentials for use with an SSL-enabled Server.

    Args:
      private_key_certificate_chain_pairs: A list of pairs of the form
        [PEM-encoded private key, PEM-encoded certificate chain].
      root_certificates: An optional byte string of PEM-encoded client root
        certificates that the server will use to verify client authentication.
        If omitted, require_client_auth must also be False.
      require_client_auth: A boolean indicating whether or not to require
        clients to be authenticated. May only be True if root_certificates
        is not None.

    Returns:
      A ServerCredentials for use with an SSL-enabled Server. Typically, this
      object is an argument to add_secure_port() method during server setup.
    """
    if not private_key_certificate_chain_pairs:
        raise ValueError('At least one private key-certificate chain pair is required!')
    elif require_client_auth and root_certificates is None:
        raise ValueError('Illegal to require client auth without providing root certificates!')
    else:
        return ServerCredentials(_cygrpc.server_credentials_ssl(root_certificates, [_cygrpc.SslPemKeyCertPair(key, pem) for key, pem in private_key_certificate_chain_pairs], require_client_auth))