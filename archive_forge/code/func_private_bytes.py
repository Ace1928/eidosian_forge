from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def private_bytes(self, encoding: serialization.Encoding, format: serialization.PrivateFormat, encryption_algorithm: serialization.KeySerializationEncryption) -> bytes:
    return self._backend._private_key_bytes(encoding, format, encryption_algorithm, self, self._evp_pkey, self._ec_key)