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
def ssh_KEX_DH_GEX_INIT(self, packet):
    """
        Called when we get a MSG_KEX_DH_GEX_INIT message.  Payload::
            integer e (client DH public key)

        We send the MSG_KEX_DH_GEX_REPLY message with our host key and
        signature.

        @type packet: L{bytes}
        @param packet: The message data.
        """
    clientDHpublicKey, foo = getMP(packet)
    pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
    sharedSecret = self._finishEphemeralDH(clientDHpublicKey)
    h = _kex.getHashProcessor(self.kexAlg)()
    h.update(NS(self.otherVersionString))
    h.update(NS(self.ourVersionString))
    h.update(NS(self.otherKexInitPayload))
    h.update(NS(self.ourKexInitPayload))
    h.update(NS(pubHostKey.blob()))
    h.update(self.dhGexRequest)
    h.update(MP(self.p))
    h.update(MP(self.g))
    h.update(MP(clientDHpublicKey))
    h.update(self.dhSecretKeyPublicMP)
    h.update(sharedSecret)
    exchangeHash = h.digest()
    self.sendPacket(MSG_KEX_DH_GEX_REPLY, NS(pubHostKey.blob()) + self.dhSecretKeyPublicMP + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
    self._keySetup(sharedSecret, exchangeHash)