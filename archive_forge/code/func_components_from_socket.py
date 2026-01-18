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
def components_from_socket(self):
    """This function returns the certificate, primary issuer, and primary ocsp
        server in the chain for a socket already wrapped with ssl.
        """
    der = self.SOCK.getpeercert(True)
    if der is False:
        raise ConnectionError('no certificate found for ssl peer')
    cert = self._bin2ascii(der)
    return self._certificate_components(cert)