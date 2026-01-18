import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse
import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from redis.exceptions import AuthorizationError, ConnectionError
def ocsp_staple_verifier(con, ocsp_bytes, expected=None):
    """An implementation of a function for set_ocsp_client_callback in PyOpenSSL.

    This function validates that the provide ocsp_bytes response is valid,
    and matches the expected, stapled responses.
    """
    if ocsp_bytes in [b'', None]:
        raise ConnectionError('no ocsp response present')
    issuer_cert = None
    peer_cert = con.get_peer_certificate().to_cryptography()
    for c in con.get_peer_cert_chain():
        cert = c.to_cryptography()
        if cert.subject == peer_cert.issuer:
            issuer_cert = cert
            break
    if issuer_cert is None:
        raise ConnectionError('no matching issuer cert found in certificate chain')
    if expected is not None:
        e = x509.load_pem_x509_certificate(expected)
        if peer_cert != e:
            raise ConnectionError('received and expected certificates do not match')
    return _check_certificate(issuer_cert, ocsp_bytes)