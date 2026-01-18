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
def load_pkcs7_data(type: int, buffer: Union[str, bytes]) -> PKCS7:
    """
    Load pkcs7 data from the string *buffer* encoded with the type
    *type*.

    :param type: The file type (one of FILETYPE_PEM or FILETYPE_ASN1)
    :param buffer: The buffer with the pkcs7 data.
    :return: The PKCS7 object
    """
    if isinstance(buffer, str):
        buffer = buffer.encode('ascii')
    bio = _new_mem_buf(buffer)
    if type == FILETYPE_PEM:
        pkcs7 = _lib.PEM_read_bio_PKCS7(bio, _ffi.NULL, _ffi.NULL, _ffi.NULL)
    elif type == FILETYPE_ASN1:
        pkcs7 = _lib.d2i_PKCS7_bio(bio, _ffi.NULL)
    else:
        raise ValueError('type argument must be FILETYPE_PEM or FILETYPE_ASN1')
    if pkcs7 == _ffi.NULL:
        _raise_current_error()
    pypkcs7 = PKCS7.__new__(PKCS7)
    pypkcs7._pkcs7 = _ffi.gc(pkcs7, _lib.PKCS7_free)
    return pypkcs7