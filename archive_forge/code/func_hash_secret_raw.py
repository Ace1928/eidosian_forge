from __future__ import annotations
from enum import Enum
from typing import Any
from _argon2_cffi_bindings import ffi, lib
from ._typing import Literal
from .exceptions import HashingError, VerificationError, VerifyMismatchError
def hash_secret_raw(secret: bytes, salt: bytes, time_cost: int, memory_cost: int, parallelism: int, hash_len: int, type: Type, version: int=ARGON2_VERSION) -> bytes:
    """
    Hash *password* and return a **raw** hash.

    This function takes the same parameters as :func:`hash_secret`.

    .. versionadded:: 16.0.0
    """
    buf = ffi.new('uint8_t[]', hash_len)
    rv = lib.argon2_hash(time_cost, memory_cost, parallelism, ffi.new('uint8_t[]', secret), len(secret), ffi.new('uint8_t[]', salt), len(salt), buf, hash_len, ffi.NULL, 0, type.value, version)
    if rv != lib.ARGON2_OK:
        raise HashingError(error_to_str(rv))
    return bytes(ffi.buffer(buf, hash_len))