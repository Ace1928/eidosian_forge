from __future__ import annotations
import binascii
import hmac
import struct
import types
import zlib
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Any, Callable, Dict, Tuple, Union
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing_extensions import Literal
from twisted import __version__ as twisted_version
from twisted.conch.ssh import _kex, address, keys
from twisted.conch.ssh.common import MP, NS, ffs, getMP, getNS
from twisted.internet import defer, protocol
from twisted.logger import Logger
from twisted.python import randbytes
from twisted.python.compat import iterbytes, networkString
def ssh_KEX_DH_GEX_REPLY(self, packet):
    """
        Called when we receive a MSG_KEX_DH_GEX_REPLY message.  Payload::
            string server host key
            integer f (server DH public key)

        We verify the host key by calling verifyHostKey, then continue in
        _continueGEX_REPLY.

        @type packet: L{bytes}
        @param packet: The message data.

        @return: A deferred firing once key exchange is complete.
        """
    pubKey, packet = getNS(packet)
    f, packet = getMP(packet)
    signature, packet = getNS(packet)
    fingerprint = b':'.join([binascii.hexlify(c) for c in iterbytes(md5(pubKey).digest())])
    d = self.verifyHostKey(pubKey, fingerprint)
    d.addCallback(self._continueGEX_REPLY, pubKey, f, signature)
    d.addErrback(lambda unused: self.sendDisconnect(DISCONNECT_HOST_KEY_NOT_VERIFIABLE, b'bad host key'))
    return d