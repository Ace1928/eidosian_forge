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
def ntowfv2(username: str, nt_hash: bytes, domain_name: typing.Optional[str]) -> bytes:
    """NTLMv2 NTOWFv2 function

    The NT v2 one way function as documented under `NTLM v2 Authentication`_.

    The pseudo-code for this function is::

        Define NTOWFv2(Passwd, User, UserDom) as

            HMAC_MD5(MD4(UNICODE(Passwd)), UNICODE(ConcatenationOf(Uppercase(User), UserDom)))

    Args:
        username: The username.
        nt_hash: The NT hash from :meth:`ntowfv1`.
        domain_name: The optional domain name of the user.

    Returns:
        bytes: The NTv2 one way has of the user's credentials.

    .. _NTLM v2 Authentication:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/5e550938-91d4-459f-b67d-75d70009e3f3
    """
    b_user = (username.upper() + (domain_name or '')).encode('utf-16-le')
    return hmac_md5(nt_hash, b_user)