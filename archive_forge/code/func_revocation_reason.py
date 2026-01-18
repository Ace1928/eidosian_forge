from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
@property
@abc.abstractmethod
def revocation_reason(self) -> typing.Optional[x509.ReasonFlags]:
    """
        The reason the certificate was revoked or None if not specified or
        not revoked.
        """