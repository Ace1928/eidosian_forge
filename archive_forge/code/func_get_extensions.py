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
def get_extensions(self) -> List[X509Extension]:
    """
        Get X.509 extensions in the certificate signing request.

        :return: The X.509 extensions in this request.
        :rtype: :py:class:`list` of :py:class:`X509Extension` objects.

        .. versionadded:: 0.15
        """
    exts = []
    native_exts_obj = _lib.X509_REQ_get_extensions(self._req)
    native_exts_obj = _ffi.gc(native_exts_obj, lambda x: _lib.sk_X509_EXTENSION_pop_free(x, _ffi.addressof(_lib._original_lib, 'X509_EXTENSION_free')))
    for i in range(_lib.sk_X509_EXTENSION_num(native_exts_obj)):
        ext = X509Extension.__new__(X509Extension)
        extension = _lib.X509_EXTENSION_dup(_lib.sk_X509_EXTENSION_value(native_exts_obj, i))
        ext._extension = _ffi.gc(extension, _lib.X509_EXTENSION_free)
        exts.append(ext)
    return exts