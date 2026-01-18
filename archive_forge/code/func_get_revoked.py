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
def get_revoked(self) -> Optional[Tuple[Revoked, ...]]:
    """
        Return the revocations in this certificate revocation list.

        These revocations will be provided by value, not by reference.
        That means it's okay to mutate them: it won't affect this CRL.

        :return: The revocations in this CRL.
        :rtype: :class:`tuple` of :class:`Revocation`
        """
    results = []
    revoked_stack = _lib.X509_CRL_get_REVOKED(self._crl)
    for i in range(_lib.sk_X509_REVOKED_num(revoked_stack)):
        revoked = _lib.sk_X509_REVOKED_value(revoked_stack, i)
        revoked_copy = _lib.X509_REVOKED_dup(revoked)
        pyrev = Revoked.__new__(Revoked)
        pyrev._revoked = _ffi.gc(revoked_copy, _lib.X509_REVOKED_free)
        results.append(pyrev)
    if results:
        return tuple(results)
    return None