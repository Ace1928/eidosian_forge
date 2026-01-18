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
def sendKexInit(self):
    """
        Send a I{KEXINIT} message to initiate key exchange or to respond to a
        key exchange initiated by the peer.

        @raise RuntimeError: If a key exchange has already been started and it
            is not appropriate to send a I{KEXINIT} message at this time.

        @return: L{None}
        """
    if self._keyExchangeState != self._KEY_EXCHANGE_NONE:
        raise RuntimeError('Cannot send KEXINIT while key exchange state is %r' % (self._keyExchangeState,))
    supportedKeyExchanges = list(self.supportedKeyExchanges)
    supportedKeyExchanges.append(self._EXT_INFO_C if self.isClient else self._EXT_INFO_S)
    self.ourKexInitPayload = b''.join([bytes((MSG_KEXINIT,)), randbytes.secureRandom(16), NS(b','.join(supportedKeyExchanges)), NS(b','.join(self.supportedPublicKeys)), NS(b','.join(self.supportedCiphers)), NS(b','.join(self.supportedCiphers)), NS(b','.join(self.supportedMACs)), NS(b','.join(self.supportedMACs)), NS(b','.join(self.supportedCompressions)), NS(b','.join(self.supportedCompressions)), NS(b','.join(self.supportedLanguages)), NS(b','.join(self.supportedLanguages)), b'\x00\x00\x00\x00\x00'])
    self.sendPacket(MSG_KEXINIT, self.ourKexInitPayload[1:])
    self._keyExchangeState = self._KEY_EXCHANGE_REQUESTED
    self._blockedByKeyExchange = []