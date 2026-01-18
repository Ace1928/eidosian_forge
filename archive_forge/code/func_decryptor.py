from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
def decryptor(self):
    from cryptography.hazmat.backends.openssl.backend import backend
    ctx = backend.create_symmetric_decryption_ctx(self.algorithm, self.mode)
    return self._wrap_ctx(ctx, encrypt=False)