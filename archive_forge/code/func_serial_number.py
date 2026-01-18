from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
def serial_number(self, number: int) -> RevokedCertificateBuilder:
    if not isinstance(number, int):
        raise TypeError('Serial number must be of integral type.')
    if self._serial_number is not None:
        raise ValueError('The serial number may only be set once.')
    if number <= 0:
        raise ValueError('The serial number should be positive')
    if number.bit_length() >= 160:
        raise ValueError('The serial number should not be more than 159 bits.')
    return RevokedCertificateBuilder(number, self._revocation_date, self._extensions)