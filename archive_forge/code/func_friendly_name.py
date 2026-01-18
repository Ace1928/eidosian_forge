from __future__ import annotations
import typing
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives._serialization import PBES as PBES
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
@property
def friendly_name(self) -> typing.Optional[bytes]:
    return self._friendly_name