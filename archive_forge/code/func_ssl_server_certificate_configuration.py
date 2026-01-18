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
def ssl_server_certificate_configuration(private_key_certificate_chain_pairs, root_certificates=None):
    """Creates a ServerCertificateConfiguration for use with a Server.

    Args:
      private_key_certificate_chain_pairs: A collection of pairs of
        the form [PEM-encoded private key, PEM-encoded certificate
        chain].
      root_certificates: An optional byte string of PEM-encoded client root
        certificates that the server will use to verify client authentication.

    Returns:
      A ServerCertificateConfiguration that can be returned in the certificate
        configuration fetching callback.
    """
    if private_key_certificate_chain_pairs:
        return ServerCertificateConfiguration(_cygrpc.server_certificate_config_ssl(root_certificates, [_cygrpc.SslPemKeyCertPair(key, pem) for key, pem in private_key_certificate_chain_pairs]))
    else:
        raise ValueError('At least one private key-certificate chain pair is required!')