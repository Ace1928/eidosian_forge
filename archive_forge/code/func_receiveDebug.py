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
def receiveDebug(self, alwaysDisplay, message, lang):
    """
        Called when we receive a debug message from the other side.

        @param alwaysDisplay: if True, this message should always be
                              displayed.
        @type alwaysDisplay: L{bool}
        @param message: the debug message
        @type message: L{str}
        @param lang: optionally the language the message is in.
        @type lang: L{str}
        """
    if alwaysDisplay:
        self._log.debug('Remote Debug Message: {message}', message=message)