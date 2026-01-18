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
def load_certificate_request(type: int, buffer: bytes) -> X509Req:
    """
    Load a certificate request (X509Req) from the string *buffer* encoded with
    the type *type*.

    :param type: The file type (one of FILETYPE_PEM, FILETYPE_ASN1)
    :param buffer: The buffer the certificate request is stored in
    :return: The X509Req object
    """
    if isinstance(buffer, str):
        buffer = buffer.encode('ascii')
    bio = _new_mem_buf(buffer)
    if type == FILETYPE_PEM:
        req = _lib.PEM_read_bio_X509_REQ(bio, _ffi.NULL, _ffi.NULL, _ffi.NULL)
    elif type == FILETYPE_ASN1:
        req = _lib.d2i_X509_REQ_bio(bio, _ffi.NULL)
    else:
        raise ValueError('type argument must be FILETYPE_PEM or FILETYPE_ASN1')
    _openssl_assert(req != _ffi.NULL)
    x509req = X509Req.__new__(X509Req)
    x509req._req = _ffi.gc(req, _lib.X509_REQ_free)
    return x509req