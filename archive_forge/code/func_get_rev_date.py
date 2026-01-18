import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
def get_rev_date(self) -> Optional[bytes]:
    """
        Get the revocation timestamp.

        :return: The timestamp of the revocation, as ASN.1 TIME.
        :rtype: bytes
        """
    dt = _lib.X509_REVOKED_get0_revocationDate(self._revoked)
    return _get_asn1_time(dt)