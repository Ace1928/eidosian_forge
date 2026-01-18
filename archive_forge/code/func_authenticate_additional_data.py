from __future__ import annotations
import typing
from cryptography.exceptions import InvalidTag, UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives import ciphers
from cryptography.hazmat.primitives.ciphers import algorithms, modes
def authenticate_additional_data(self, data: bytes) -> None:
    outlen = self._backend._ffi.new('int *')
    res = self._backend._lib.EVP_CipherUpdate(self._ctx, self._backend._ffi.NULL, outlen, self._backend._ffi.from_buffer(data), len(data))
    self._backend.openssl_assert(res != 0)