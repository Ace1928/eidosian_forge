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
def set_rev_date(self, when: bytes) -> None:
    """
        Set the revocation timestamp.

        :param bytes when: The timestamp of the revocation,
            as ASN.1 TIME.
        :return: ``None``
        """
    revocationDate = _new_asn1_time(when)
    ret = _lib.X509_REVOKED_set_revocationDate(self._revoked, revocationDate)
    _openssl_assert(ret == 1)