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
def b64_encode(self) -> bytes:
    """
        Generate a base64 encoded representation of this SPKI object.

        :return: The base64 encoded string.
        :rtype: :py:class:`bytes`
        """
    encoded = _lib.NETSCAPE_SPKI_b64_encode(self._spki)
    result = _ffi.string(encoded)
    _lib.OPENSSL_free(encoded)
    return result