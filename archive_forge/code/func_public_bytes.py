from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def public_bytes(self, encoding: serialization.Encoding, format: serialization.PublicFormat) -> bytes:
    if encoding is serialization.Encoding.X962 or format is serialization.PublicFormat.CompressedPoint or format is serialization.PublicFormat.UncompressedPoint:
        if encoding is not serialization.Encoding.X962 or format not in (serialization.PublicFormat.CompressedPoint, serialization.PublicFormat.UncompressedPoint):
            raise ValueError('X962 encoding must be used with CompressedPoint or UncompressedPoint format')
        return self._encode_point(format)
    else:
        return self._backend._public_key_bytes(encoding, format, self, self._evp_pkey, None)