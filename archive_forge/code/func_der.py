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
def der(self) -> bytes:
    """
        Return the DER encoding of this name.

        :return: The DER encoded form of this name.
        :rtype: :py:class:`bytes`
        """
    result_buffer = _ffi.new('unsigned char**')
    encode_result = _lib.i2d_X509_NAME(self._name, result_buffer)
    _openssl_assert(encode_result >= 0)
    string_result = _ffi.buffer(result_buffer[0], encode_result)[:]
    _lib.OPENSSL_free(result_buffer[0])
    return string_result