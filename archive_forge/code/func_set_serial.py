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
def set_serial(self, hex_str: bytes) -> None:
    """
        Set the serial number.

        The serial number is formatted as a hexadecimal number encoded in
        ASCII.

        :param bytes hex_str: The new serial number.

        :return: ``None``
        """
    bignum_serial = _ffi.gc(_lib.BN_new(), _lib.BN_free)
    bignum_ptr = _ffi.new('BIGNUM**')
    bignum_ptr[0] = bignum_serial
    bn_result = _lib.BN_hex2bn(bignum_ptr, hex_str)
    if not bn_result:
        raise ValueError('bad hex string')
    asn1_serial = _ffi.gc(_lib.BN_to_ASN1_INTEGER(bignum_serial, _ffi.NULL), _lib.ASN1_INTEGER_free)
    _lib.X509_REVOKED_set_serialNumber(self._revoked, asn1_serial)