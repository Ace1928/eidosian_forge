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
def rc4k(k: bytes, d: bytes) -> bytes:
    """RC4 encryption with an explicit key.

    Indicates the encryption of data item `d` with the key `k` using the `RC4`_ algorithm.

    Args:
        k: The key to use for the RC4 cipher.
        d: The data to encrypt.

    Returns:
        bytes: The RC4 encrypted bytes.

    .. _RC4K:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/26c42637-9549-46ae-be2e-90f6f1360193
    """
    return rc4init(k).update(d)