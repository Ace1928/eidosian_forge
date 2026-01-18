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
def signkey(flags: int, session_key: bytes, usage: str) -> bytes:
    """NTLM SIGNKEY function.

    The MS-NLMP `SIGNKEY`_ function used to generate the signing keys for a security context.

    The pseudo-code for this function as documented under `SIGNKEY`_ is::

        Define SIGNKEY(NegFlg, ExportedSessionKey, Mode) as

            If (NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY flag is set in NegFlg)
                If (Mode equals "Client")
                    Set SignKey to MD5(ConcatenationOf(ExportedSessionKey,
                        "session key to client-to-server signing key magic constant"))

                Else
                    Set SignKey to MD5(ConcatenationOf(ExportedSessionKey,
                        "session key to server-to-client signing key magic constant"))

                Endif
            Else
                Set  SignKey to NIL

            Endif
        EndDefine

    Args:
        flags: The negotiated flags between the initiator and acceptor.
        session_key: The derived session key.
        usage: Whether the signing key is for the 'initiate' or 'accept' context.

    Returns:
        bytes: The derived singing key.

    .. _SIGNKEY:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/524cdccb-563e-4793-92b0-7bc321fce096
    """
    if flags & NegotiateFlags.extended_session_security == 0:
        return b''
    direction = b'client-to-server' if usage == 'initiate' else b'server-to-client'
    return md5(session_key + b'session key to %s signing key magic constant\x00' % direction)