from __future__ import annotations
from enum import Enum
from typing import Any
from _argon2_cffi_bindings import ffi, lib
from ._typing import Literal
from .exceptions import HashingError, VerificationError, VerifyMismatchError
def hash_secret(secret: bytes, salt: bytes, time_cost: int, memory_cost: int, parallelism: int, hash_len: int, type: Type, version: int=ARGON2_VERSION) -> bytes:
    """
    Hash *secret* and return an **encoded** hash.

    An encoded hash can be directly passed into :func:`verify_secret` as it
    contains all parameters and the salt.

    :param bytes secret: Secret to hash.
    :param bytes salt: A salt_.  Should be random and different for each
        secret.
    :param Type type: Which Argon2 variant to use.
    :param int version: Which Argon2 version to use.

    For an explanation of the Argon2 parameters see
    :class:`argon2.PasswordHasher`.

    :rtype: bytes

    :raises argon2.exceptions.HashingError: If hashing fails.

    .. versionadded:: 16.0.0

    .. _salt: https://en.wikipedia.org/wiki/Salt_(cryptography)
    .. _kibibytes: https://en.wikipedia.org/wiki/Binary_prefix#kibi
    """
    size = lib.argon2_encodedlen(time_cost, memory_cost, parallelism, len(salt), hash_len, type.value) + 1
    buf = ffi.new('char[]', size)
    rv = lib.argon2_hash(time_cost, memory_cost, parallelism, ffi.new('uint8_t[]', secret), len(secret), ffi.new('uint8_t[]', salt), len(salt), ffi.NULL, hash_len, buf, size, type.value, version)
    if rv != lib.ARGON2_OK:
        raise HashingError(error_to_str(rv))
    return ffi.string(buf)