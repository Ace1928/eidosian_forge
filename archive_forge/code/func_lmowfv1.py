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
def lmowfv1(password: str) -> bytes:
    """NTLMv1 LMOWFv1 function

    The Lan Manager v1 one way function as documented under `NTLM v1 Authentication`_.

    The pseudo-code for this function is::

        Define LMOWFv1(Passwd, User, UserDom) as
            ConcatenationOf(
                DES(UpperCase(Passwd)[0..6], "KGS!@#$%"),
                DES(UpperCase(Passwd)[7..13], "KGS!@#$%"),
            );

    Args:
        password: The password for the user.

    Returns:
        bytes: The LMv1 one way hash of the user's password.

    .. _NTLM v1 Authentication:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/464551a8-9fc4-428e-b3d3-bc5bfb2e73a5
    """
    if is_ntlm_hash(password):
        return base64.b16decode(password.split(':')[0].upper())
    b_password = password.upper().encode('utf-8').ljust(14, b'\x00')[:14]
    b_hash = io.BytesIO()
    for start, end in [(0, 7), (7, 14)]:
        b_hash.write(des(b_password[start:end], b'KGS!@#$%'))
    return b_hash.getvalue()