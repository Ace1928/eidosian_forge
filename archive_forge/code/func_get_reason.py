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
def get_reason(self) -> Optional[bytes]:
    """
        Get the reason of this revocation.

        :return: The reason, or ``None`` if there is none.
        :rtype: bytes or NoneType

        .. seealso::

            :meth:`all_reasons`, which gives you a list of all supported
            reasons this method might return.
        """
    for i in range(_lib.X509_REVOKED_get_ext_count(self._revoked)):
        ext = _lib.X509_REVOKED_get_ext(self._revoked, i)
        obj = _lib.X509_EXTENSION_get_object(ext)
        if _lib.OBJ_obj2nid(obj) == _lib.NID_crl_reason:
            bio = _new_mem_buf()
            print_result = _lib.X509V3_EXT_print(bio, ext, 0, 0)
            if not print_result:
                print_result = _lib.M_ASN1_OCTET_STRING_print(bio, _lib.X509_EXTENSION_get_data(ext))
                _openssl_assert(print_result != 0)
            return _bio_to_string(bio)
    return None