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
def check_certificate(self, server, cert, issuer_url):
    """Checks the validity of an ocsp server for an issuer"""
    r = requests.get(issuer_url)
    if not r.ok:
        raise ConnectionError('failed to fetch issuer certificate')
    der = r.content
    issuer_cert = self._bin2ascii(der)
    ocsp_url = self.build_certificate_url(server, cert, issuer_cert)
    header = {'Host': urlparse(ocsp_url).netloc, 'Content-Type': 'application/ocsp-request'}
    r = requests.get(ocsp_url, headers=header)
    if not r.ok:
        raise ConnectionError('failed to fetch ocsp certificate')
    return _check_certificate(issuer_cert, r.content, True)