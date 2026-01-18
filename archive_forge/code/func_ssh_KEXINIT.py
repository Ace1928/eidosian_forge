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
def ssh_KEXINIT(self, packet):
    """
        Called when we receive a MSG_KEXINIT message.  For a description
        of the packet, see SSHTransportBase.ssh_KEXINIT().  Additionally,
        this method sends the first key exchange packet.

        If the agreed-upon exchange is ECDH, generate a key pair for the
        corresponding curve and send the public key.

        If the agreed-upon exchange has a fixed prime/generator group,
        generate a public key and send it in a MSG_KEXDH_INIT message.
        Otherwise, ask for a 2048 bit group with a MSG_KEX_DH_GEX_REQUEST
        message.
        """
    if SSHTransportBase.ssh_KEXINIT(self, packet) is None:
        return
    if _kex.isEllipticCurve(self.kexAlg):
        self.ecPriv = self._generateECPrivateKey()
        self.ecPub = self.ecPriv.public_key()
        self.sendPacket(MSG_KEX_DH_GEX_REQUEST_OLD, NS(self._encodeECPublicKey(self.ecPub)))
    elif _kex.isFixedGroup(self.kexAlg):
        self.g, self.p = _kex.getDHGeneratorAndPrime(self.kexAlg)
        self._startEphemeralDH()
        self.sendPacket(MSG_KEXDH_INIT, self.dhSecretKeyPublicMP)
    else:
        self.sendPacket(MSG_KEX_DH_GEX_REQUEST, struct.pack('!LLL', self._dhMinimalGroupSize, self._dhPreferredGroupSize, self._dhMaximalGroupSize))