from OpenSSL.crypto import PKey, X509
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (load_pem_private_key,
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.backends import default_backend
from datetime import datetime
from requests.adapters import HTTPAdapter
import requests
from .. import exceptions as exc
importing the protocol constants from _ssl instead of ssl because only the
def create_ssl_context(cert_byes, pk_bytes, password=None, encoding=Encoding.PEM):
    """Create an SSL Context with the supplied cert/password.

    :param cert_bytes array of bytes containing the cert encoded
           using the method supplied in the ``encoding`` parameter
    :param pk_bytes array of bytes containing the private key encoded
           using the method supplied in the ``encoding`` parameter
    :param password array of bytes containing the passphrase to be used
           with the supplied private key. None if unencrypted.
           Defaults to None.
    :param encoding ``cryptography.hazmat.primitives.serialization.Encoding``
            details the encoding method used on the ``cert_bytes``  and
            ``pk_bytes`` parameters. Can be either PEM or DER.
            Defaults to PEM.
    """
    backend = default_backend()
    cert = None
    key = None
    if encoding == Encoding.PEM:
        cert = x509.load_pem_x509_certificate(cert_byes, backend)
        key = load_pem_private_key(pk_bytes, password, backend)
    elif encoding == Encoding.DER:
        cert = x509.load_der_x509_certificate(cert_byes, backend)
        key = load_der_private_key(pk_bytes, password, backend)
    else:
        raise ValueError('Invalid encoding provided: Must be PEM or DER')
    if not (cert and key):
        raise ValueError('Cert and key could not be parsed from provided data')
    check_cert_dates(cert)
    ssl_context = PyOpenSSLContext(PROTOCOL)
    ssl_context._ctx.use_certificate(X509.from_cryptography(cert))
    ssl_context._ctx.use_privatekey(PKey.from_cryptography_key(key))
    return ssl_context