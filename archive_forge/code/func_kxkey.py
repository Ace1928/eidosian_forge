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
def kxkey(flags: int, session_base_key: bytes, lmowf: bytes, lm_response: bytes, server_challenge: bytes) -> bytes:
    """NTLM KXKEY function.

    The MS-NLMP `KXKEY`_ function used to derive the key exchange key for a security context. This is only for NTLMv1
    contexts as NTLMv2 just re-uses the session base key.

    Args:
        flags: The negotiate flags in the Challenge msg.
        session_base_key: The session base key from :meth:`compute_response_v1`.
        lmowf: The LM hash from :meth:`lmowfv1`.
        lm_response: The lm response from :meth:`compute_response_v1`.
        server_challenge: The server challenge in the Challenge msg.

    Returns:
        bytes: The derived key exchange key.

    .. _KXKEY:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/d86303b5-b29e-4fb9-b119-77579c761370
    """
    if flags & NegotiateFlags.extended_session_security:
        return hmac_md5(session_base_key, server_challenge + lm_response[:8])
    elif flags & NegotiateFlags.lm_key:
        b_data = lm_response[:8]
        return des(lmowf[:7], b_data) + des(lmowf[7:8] + b'\xbd\xbd\xbd\xbd\xbd\xbd', b_data)
    elif flags & NegotiateFlags.non_nt_session_key:
        return lmowf[:8] + b'\x00' * 8
    else:
        return session_base_key