from __future__ import annotations
import abc
import datetime
import os
import typing
import warnings
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
@property
@abc.abstractmethod
def not_valid_after_utc(self) -> datetime.datetime:
    """
        Not after time (represented as a non-naive UTC datetime)
        """