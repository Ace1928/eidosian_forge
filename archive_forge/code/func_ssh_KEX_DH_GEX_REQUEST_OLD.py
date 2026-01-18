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
def ssh_KEX_DH_GEX_REQUEST_OLD(self, packet):
    """
        This represents different key exchange methods that share the same
        integer value.  If the message is determined to be a KEXDH_INIT,
        L{_ssh_KEXDH_INIT} is called to handle it. If it is a KEX_ECDH_INIT,
        L{_ssh_KEX_ECDH_INIT} is called.
        Otherwise, for KEX_DH_GEX_REQUEST_OLD payload::

                integer ideal (ideal size for the Diffie-Hellman prime)

            We send the KEX_DH_GEX_GROUP message with the group that is
            closest in size to ideal.

        If we were told to ignore the next key exchange packet by ssh_KEXINIT,
        drop it on the floor and return.

        @type packet: L{bytes}
        @param packet: The message data.
        """
    if self.ignoreNextPacket:
        self.ignoreNextPacket = 0
        return
    if _kex.isFixedGroup(self.kexAlg):
        return self._ssh_KEXDH_INIT(packet)
    elif _kex.isEllipticCurve(self.kexAlg):
        return self._ssh_KEX_ECDH_INIT(packet)
    else:
        self.dhGexRequest = packet
        ideal = struct.unpack('>L', packet)[0]
        self.g, self.p = self.factory.getDHPrime(ideal)
        self._startEphemeralDH()
        self.sendPacket(MSG_KEX_DH_GEX_GROUP, MP(self.p) + MP(self.g))