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
def not_valid_after(self, time: datetime.datetime) -> CertificateBuilder:
    """
        Sets the certificate expiration time.
        """
    if not isinstance(time, datetime.datetime):
        raise TypeError('Expecting datetime object.')
    if self._not_valid_after is not None:
        raise ValueError('The not valid after may only be set once.')
    time = _convert_to_naive_utc_time(time)
    if time < _EARLIEST_UTC_TIME:
        raise ValueError('The not valid after date must be on or after 1950 January 1.')
    if self._not_valid_before is not None and time < self._not_valid_before:
        raise ValueError('The not valid after date must be after the not valid before date.')
    return CertificateBuilder(self._issuer_name, self._subject_name, self._public_key, self._serial_number, self._not_valid_before, time, self._extensions)