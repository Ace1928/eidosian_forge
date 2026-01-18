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
def dynamic_ssl_server_credentials(initial_certificate_configuration, certificate_configuration_fetcher, require_client_authentication=False):
    """Creates a ServerCredentials for use with an SSL-enabled Server.

    Args:
      initial_certificate_configuration (ServerCertificateConfiguration): The
        certificate configuration with which the server will be initialized.
      certificate_configuration_fetcher (callable): A callable that takes no
        arguments and should return a ServerCertificateConfiguration to
        replace the server's current certificate, or None for no change
        (i.e., the server will continue its current certificate
        config). The library will call this callback on *every* new
        client connection before starting the TLS handshake with the
        client, thus allowing the user application to optionally
        return a new ServerCertificateConfiguration that the server will then
        use for the handshake.
      require_client_authentication: A boolean indicating whether or not to
        require clients to be authenticated.

    Returns:
      A ServerCredentials.
    """
    return ServerCredentials(_cygrpc.server_credentials_ssl_dynamic_cert_config(initial_certificate_configuration, certificate_configuration_fetcher, require_client_authentication))