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
def sealkey(flags: int, session_key: bytes, usage: str) -> bytes:
    """NTLM SEALKEY function.

    The MS-NLMP `SEALKEY`_ function used to generate the sealing keys for a security context.

    The pseudo-code for this function as documented under `SEALKEY`_ is::

        Define SEALKEY(NegFlg, ExportedSessionKey, Mode) as

            If (NTLMSSP_NEGOTIATE_EXTENDED_SESSIONSECURITY flag is set in NegFlg)

                If ( NTLMSSP_NEGOTIATE_128 is set in NegFlg)
                    Set SealKey to ExportedSessionKey

                ElseIf ( NTLMSSP_NEGOTIATE_56 flag is set in NegFlg)
                    Set SealKey to ExportedSessionKey[0..6]

                Else
                    Set SealKey to ExportedSessionKey[0..4]

                Endif

                If (Mode equals "Client")
                    Set SealKey to MD5(ConcatenationOf(SealKey,
                        "session key to client-to-server sealing key magic constant"))

                Else
                    Set SealKey to MD5(ConcatenationOf(SealKey,
                        "session key to server-to-client sealing key magic constant"))

                Endif
            ElseIf ((NTLMSSP_NEGOTIATE_LM_KEY is set in NegFlg) or ((NTLMSSP_NEGOTIATE_DATAGRAM is set in NegFlg) and
                                                                    (NTLMRevisionCurrent >= NTLMSSP_REVISION_W2K3)))

                If (NTLMSSP_NEGOTIATE_56 flag is set in NegFlg)
                    Set SealKey to ConcatenationOf(ExportedSessionKey[0..6], 0xA0)

                Else
                    Set SealKey to ConcatenationOf(ExportedSessionKey[0..4], 0xE5, 0x38, 0xB0)

                EndIf

            Else
                Set SealKey to ExportedSessionKey
            Endif
        EndDefine

    Args:
        flags: The negotiated flags between the initiator and acceptor.
        session_key: The derived session key.
        usage: Whether the sealing key is for the 'initiate' or 'accept' context.

    Returns:
        bytes: The derived sealing key.

    .. _SEALKEY:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/bf39181d-e95d-40d7-a740-ab4ec3dc363d
    """
    if flags & NegotiateFlags.extended_session_security:
        if flags & NegotiateFlags.key_128:
            seal_key = session_key
        elif flags & NegotiateFlags.key_56:
            seal_key = session_key[:7]
        else:
            seal_key = session_key[:5]
        direction = b'client-to-server' if usage == 'initiate' else b'server-to-client'
        return md5(seal_key + b'session key to %s sealing key magic constant\x00' % direction)
    elif flags & NegotiateFlags.lm_key or flags & NegotiateFlags.datagram:
        if flags & NegotiateFlags.key_56:
            return session_key[:7] + b'\xa0'
        else:
            return session_key[:5] + b'\xe58\xb0'
    else:
        return session_key