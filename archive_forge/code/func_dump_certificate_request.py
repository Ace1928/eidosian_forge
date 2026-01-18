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
def dump_certificate_request(type: int, req: X509Req) -> bytes:
    """
    Dump the certificate request *req* into a buffer string encoded with the
    type *type*.

    :param type: The file type (one of FILETYPE_PEM, FILETYPE_ASN1)
    :param req: The certificate request to dump
    :return: The buffer with the dumped certificate request in
    """
    bio = _new_mem_buf()
    if type == FILETYPE_PEM:
        result_code = _lib.PEM_write_bio_X509_REQ(bio, req._req)
    elif type == FILETYPE_ASN1:
        result_code = _lib.i2d_X509_REQ_bio(bio, req._req)
    elif type == FILETYPE_TEXT:
        result_code = _lib.X509_REQ_print_ex(bio, req._req, 0, 0)
    else:
        raise ValueError('type argument must be FILETYPE_PEM, FILETYPE_ASN1, or FILETYPE_TEXT')
    _openssl_assert(result_code != 0)
    return _bio_to_string(bio)