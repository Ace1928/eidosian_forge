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
def get_short_name(self) -> bytes:
    """
        Returns the short type name of this X.509 extension.

        The result is a byte string such as :py:const:`b"basicConstraints"`.

        :return: The short type name.
        :rtype: :py:data:`bytes`

        .. versionadded:: 0.12
        """
    obj = _lib.X509_EXTENSION_get_object(self._extension)
    nid = _lib.OBJ_obj2nid(obj)
    buf = _lib.OBJ_nid2sn(nid)
    if buf != _ffi.NULL:
        return _ffi.string(buf)
    else:
        return b'UNDEF'