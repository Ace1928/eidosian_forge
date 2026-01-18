import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def generate_csr_and_key():
    """Return a dict with a new csr and key."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    csr = x509.CertificateSigningRequestBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u'admin'), x509.NameAttribute(NameOID.ORGANIZATION_NAME, u'system:masters')])).sign(key, hashes.SHA256(), default_backend())
    result = {'csr': csr.public_bytes(encoding=serialization.Encoding.PEM).decode('utf-8'), 'key': key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.TraditionalOpenSSL, encryption_algorithm=serialization.NoEncryption()).decode('utf-8')}
    return result