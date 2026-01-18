import base64
import binascii
import hashlib
import hmac
import io
import re
import struct
import typing
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from spnego._ntlm_raw.des import DES
from spnego._ntlm_raw.md4 import md4
from spnego._ntlm_raw.messages import (
def desl(k: bytes, d: bytes) -> bytes:
    """Encryption using the DES Long algorithm.

    Indicates the encryption of an 8-byte data item `d` with the 16-byte key `k` using the Data Encryption
    Standard Long (DESL) algorithm. The result is 24 bytes in length.

    `DESL(K, D)` as by MS-NLMP `DESL`_ is computed as follows::

        ConcatenationOf(
            DES(K[0..6], D),
            DES(K[7..13], D),
            DES(ConcatenationOf(K[14..15], Z(5)), D),
        );

    Args:
        k: The key to use for the DES cipher, will be truncated to 16 bytes and then padded to 21 bytes.
        d: The value to run through the DESL algorithm, will be truncated to 8 bytes.

    Returns:
        bytes: The output of the DESL algorithm.

    .. _DESL:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/26c42637-9549-46ae-be2e-90f6f1360193
    """
    k = k[:16].ljust(21, b'\x00')
    d = d[:8].ljust(8, b'\x00')
    b_value = io.BytesIO()
    b_value.write(des(k[:7], d))
    b_value.write(des(k[7:14], d))
    b_value.write(des(k[14:], d))
    return b_value.getvalue()