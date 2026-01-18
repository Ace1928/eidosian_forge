import dataclasses
import datetime
import platform
import ssl
import typing
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
def get_certificate_public_key(data: bytes) -> bytes:
    """Public key bytes of an X.509 Certificate.

    Gets the public key bytes used by CredSSP of the provided X.509
    certificate. Use this for the `public_key` attribute of
    class:`CredSSPTLSContext` for an acceptor context when providing your own
    certificate.

    Args:
        data: The DER encoded bytes of the X.509 certificate.

    Returns:
        bytes: The public key bytes of the certificate.
    """
    cert = x509.load_der_x509_certificate(data, default_backend())
    public_key = cert.public_key()
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.PKCS1)