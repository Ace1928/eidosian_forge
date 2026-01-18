import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def compute_O_value(R: int, password: bytes, key: bytes, u_value: bytes) -> Tuple[bytes, bytes]:
    """
        Algorithm 3.9 Computing the encryption dictionary’s O (owner password)
        and OE (owner encryption key) values.

        1. Generate 16 random bytes of data using a strong random number
           generator. The first 8 bytes are the Owner Validation Salt. The
           second 8 bytes are the Owner Key Salt. Compute the 32-byte SHA-256
           hash of the password concatenated with the Owner Validation Salt and
           then concatenated with the 48-byte U string as generated in
           Algorithm 3.8. The 48-byte string consisting of the 32-byte hash
           followed by the Owner Validation Salt followed by the Owner Key Salt
           is stored as the O key.
        2. Compute the 32-byte SHA-256 hash of the password concatenated with
           the Owner Key Salt and then concatenated with the 48-byte U string as
           generated in Algorithm 3.8. Using this hash as the key,
           encrypt the file encryption key using AES-256 in CBC mode with
           no padding and an initialization vector of zero.
           The resulting 32-byte string is stored as the OE key.

        Args:
            R:
            password:
            key:
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password
                and, if so, whether a valid user or owner password was entered.

        Returns:
            A tuple (O value, OE value)
        """
    random_bytes = secrets.token_bytes(16)
    val_salt = random_bytes[:8]
    key_salt = random_bytes[8:]
    o_value = AlgV5.calculate_hash(R, password, val_salt, u_value) + val_salt + key_salt
    tmp_key = AlgV5.calculate_hash(R, password, key_salt, u_value[:48])
    iv = bytes((0 for _ in range(16)))
    oe_value = aes_cbc_encrypt(tmp_key, iv, key)
    return (o_value, oe_value)