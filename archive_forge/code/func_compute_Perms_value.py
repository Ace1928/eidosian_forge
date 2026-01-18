import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
@staticmethod
def compute_Perms_value(key: bytes, p: int, metadata_encrypted: bool) -> bytes:
    """
        Algorithm 3.10 Computing the encryption dictionary’s Perms
        (permissions) value.

        1. Extend the permissions (contents of the P integer) to 64 bits by
           setting the upper 32 bits to all 1’s.
           (This allows for future extension without changing the format.)
        2. Record the 8 bytes of permission in the bytes 0-7 of the block,
           low order byte first.
        3. Set byte 8 to the ASCII value ' T ' or ' F ' according to the
           EncryptMetadata Boolean.
        4. Set bytes 9-11 to the ASCII characters ' a ', ' d ', ' b '.
        5. Set bytes 12-15 to 4 bytes of random data, which will be ignored.
        6. Encrypt the 16-byte block using AES-256 in ECB mode with an
           initialization vector of zero, using the file encryption key as the
           key. The result (16 bytes) is stored as the Perms string, and checked
           for validity when the file is opened.

        Args:
            key:
            p: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            The perms value
        """
    b8 = b'T' if metadata_encrypted else b'F'
    rr = secrets.token_bytes(4)
    data = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb' + rr
    perms = aes_ecb_encrypt(key, data)
    return perms