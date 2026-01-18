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
def check_cert_dates(cert):
    """Verify that the supplied client cert is not invalid."""
    now = datetime.utcnow()
    if cert.not_valid_after < now or cert.not_valid_before > now:
        raise ValueError('Client certificate expired: Not After: {:%Y-%m-%d %H:%M:%SZ} Not Before: {:%Y-%m-%d %H:%M:%SZ}'.format(cert.not_valid_after, cert.not_valid_before))