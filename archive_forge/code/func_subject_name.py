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
def subject_name(self, name: Name) -> CertificateBuilder:
    """
        Sets the requestor's distinguished name.
        """
    if not isinstance(name, Name):
        raise TypeError('Expecting x509.Name object.')
    if self._subject_name is not None:
        raise ValueError('The subject name may only be set once.')
    return CertificateBuilder(self._issuer_name, name, self._public_key, self._serial_number, self._not_valid_before, self._not_valid_after, self._extensions)