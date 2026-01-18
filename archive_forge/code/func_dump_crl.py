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
def dump_crl(type: int, crl: CRL) -> bytes:
    """
    Dump a certificate revocation list to a buffer.

    :param type: The file type (one of ``FILETYPE_PEM``, ``FILETYPE_ASN1``, or
        ``FILETYPE_TEXT``).
    :param CRL crl: The CRL to dump.

    :return: The buffer with the CRL.
    :rtype: bytes
    """
    bio = _new_mem_buf()
    if type == FILETYPE_PEM:
        ret = _lib.PEM_write_bio_X509_CRL(bio, crl._crl)
    elif type == FILETYPE_ASN1:
        ret = _lib.i2d_X509_CRL_bio(bio, crl._crl)
    elif type == FILETYPE_TEXT:
        ret = _lib.X509_CRL_print(bio, crl._crl)
    else:
        raise ValueError('type argument must be FILETYPE_PEM, FILETYPE_ASN1, or FILETYPE_TEXT')
    _openssl_assert(ret == 1)
    return _bio_to_string(bio)