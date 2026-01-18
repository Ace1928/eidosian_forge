import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def compute_U_value(R: int, password: bytes, key: bytes) -> Tuple[bytes, bytes]:
    """
        Algorithm 3.8 Computing the encryption dictionary’s U (user password)
        and UE (user encryption key) values.

        1. Generate 16 random bytes of data using a strong random number generator.
           The first 8 bytes are the User Validation Salt. The second 8 bytes
           are the User Key Salt. Compute the 32-byte SHA-256 hash of the
           password concatenated with the User Validation Salt. The 48-byte
           string consisting of the 32-byte hash followed by the User
           Validation Salt followed by the User Key Salt is stored as the U key.
        2. Compute the 32-byte SHA-256 hash of the password concatenated with
           the User Key Salt. Using this hash as the key, encrypt the file
           encryption key using AES-256 in CBC mode with no padding and an
           initialization vector of zero. The resulting 32-byte string is stored
           as the UE key.

        Args:
            R:
            password:
            key:

        Returns:
            A tuple (u-value, ue value)
        """
    random_bytes = secrets.token_bytes(16)
    val_salt = random_bytes[:8]
    key_salt = random_bytes[8:]
    u_value = AlgV5.calculate_hash(R, password, val_salt, b'') + val_salt + key_salt
    tmp_key = AlgV5.calculate_hash(R, password, key_salt, b'')
    iv = bytes((0 for _ in range(16)))
    ue_value = aes_cbc_encrypt(tmp_key, iv, key)
    return (u_value, ue_value)