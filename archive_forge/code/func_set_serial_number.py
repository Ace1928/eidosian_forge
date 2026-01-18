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
def set_serial_number(self, serial: int) -> None:
    """
        Set the serial number of the certificate.

        :param serial: The new serial number.
        :type serial: :py:class:`int`

        :return: :py:data`None`
        """
    if not isinstance(serial, int):
        raise TypeError('serial must be an integer')
    hex_serial = hex(serial)[2:]
    hex_serial_bytes = hex_serial.encode('ascii')
    bignum_serial = _ffi.new('BIGNUM**')
    small_serial = _lib.BN_hex2bn(bignum_serial, hex_serial_bytes)
    if bignum_serial[0] == _ffi.NULL:
        set_result = _lib.ASN1_INTEGER_set(_lib.X509_get_serialNumber(self._x509), small_serial)
        if set_result:
            _raise_current_error()
    else:
        asn1_serial = _lib.BN_to_ASN1_INTEGER(bignum_serial[0], _ffi.NULL)
        _lib.BN_free(bignum_serial[0])
        if asn1_serial == _ffi.NULL:
            _raise_current_error()
        asn1_serial = _ffi.gc(asn1_serial, _lib.ASN1_INTEGER_free)
        set_result = _lib.X509_set_serialNumber(self._x509, asn1_serial)
        _openssl_assert(set_result == 1)