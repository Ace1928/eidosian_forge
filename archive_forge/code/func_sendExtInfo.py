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
def sendExtInfo(self, extensions):
    """
        Send an RFC 8308 extension advertisement to the remote peer.

        Nothing is sent if the peer doesn't support negotiations.
        @type extensions: L{list} of (L{bytes}, L{bytes})
        @param extensions: a list of (extension-name, extension-value) pairs.
        """
    if self._peerSupportsExtensions:
        payload = b''.join([struct.pack('>L', len(extensions))] + [NS(name) + NS(value) for name, value in extensions])
        self.sendPacket(MSG_EXT_INFO, payload)