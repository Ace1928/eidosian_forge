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
def responder_key_hash(self) -> typing.Optional[bytes]:
    """
        The responder's key hash or None
        """