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
def last_update(self, last_update: datetime.datetime) -> CertificateRevocationListBuilder:
    if not isinstance(last_update, datetime.datetime):
        raise TypeError('Expecting datetime object.')
    if self._last_update is not None:
        raise ValueError('Last update may only be set once.')
    last_update = _convert_to_naive_utc_time(last_update)
    if last_update < _EARLIEST_UTC_TIME:
        raise ValueError('The last update date must be on or after 1950 January 1.')
    if self._next_update is not None and last_update > self._next_update:
        raise ValueError('The last update date must be before the next update date.')
    return CertificateRevocationListBuilder(self._issuer_name, last_update, self._next_update, self._extensions, self._revoked_certificates)