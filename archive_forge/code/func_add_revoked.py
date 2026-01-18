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
def add_revoked(self, revoked: Revoked) -> None:
    """
        Add a revoked (by value not reference) to the CRL structure

        This revocation will be added by value, not by reference. That
        means it's okay to mutate it after adding: it won't affect
        this CRL.

        :param Revoked revoked: The new revocation.
        :return: ``None``
        """
    copy = _lib.X509_REVOKED_dup(revoked._revoked)
    _openssl_assert(copy != _ffi.NULL)
    add_result = _lib.X509_CRL_add0_revoked(self._crl, copy)
    _openssl_assert(add_result != 0)