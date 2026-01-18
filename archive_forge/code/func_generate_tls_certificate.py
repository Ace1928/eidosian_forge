import dataclasses
import datetime
import platform
import ssl
import typing
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
def generate_tls_certificate() -> typing.Tuple[bytes, bytes, bytes]:
    """Generates X509 cert and key for CredSSP acceptor.

    Generates a TLS X509 certificate and key that can be used by a CredSSP
    acceptor for authentication. This certificate is modelled after the one
    that the WSMan CredSSP service uses on Windows.

    Returns:
        Tuple[bytes, bytes, bytes]: The X509 PEM encoded certificate,
        PEM encoded key, and DER encoded public key.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, 'CREDSSP-%s' % platform.node())])
    now = datetime.datetime.utcnow()
    cert = x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=365)).sign(key, hashes.SHA256(), default_backend())
    cert_pem = cert.public_bytes(encoding=serialization.Encoding.PEM)
    key_pem = key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption())
    public_key = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.PKCS1)
    return (cert_pem, key_pem, public_key)