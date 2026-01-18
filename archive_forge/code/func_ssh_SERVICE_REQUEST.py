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
def ssh_SERVICE_REQUEST(self, packet):
    """
        Called when we get a MSG_SERVICE_REQUEST message.  Payload::
            string serviceName

        The client has requested a service.  If we can start the service,
        start it; otherwise, disconnect with
        DISCONNECT_SERVICE_NOT_AVAILABLE.

        @type packet: L{bytes}
        @param packet: The message data.
        """
    service, rest = getNS(packet)
    cls = self.factory.getService(self, service)
    if not cls:
        self.sendDisconnect(DISCONNECT_SERVICE_NOT_AVAILABLE, b"don't have service " + service)
        return
    else:
        self.sendPacket(MSG_SERVICE_ACCEPT, NS(service))
        self.setService(cls())