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
def dispatchMessage(self, messageNum, payload):
    """
        Send a received message to the appropriate method.

        @type messageNum: L{int}
        @param messageNum: The message number.

        @type payload: L{bytes}
        @param payload: The message payload.
        """
    if messageNum < 50 and messageNum in messages:
        messageType = messages[messageNum][4:]
        f = getattr(self, f'ssh_{messageType}', None)
        if f is not None:
            f(payload)
        else:
            self._log.debug("couldn't handle {messageType}: {payload!r}", messageType=messageType, payload=payload)
            self.sendUnimplemented()
    elif self.service:
        self.service.packetReceived(messageNum, payload)
    else:
        self._log.debug("couldn't handle {messageNum}: {payload!r}", messageNum=messageNum, payload=payload)
        self.sendUnimplemented()