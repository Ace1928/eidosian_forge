import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_disconnectKEXDH_REPLYBadSignature(self):
    """
        Test that KEX_ECDH_REPLY disconnects if the signature is bad.
        """
    exchangeHash, signature, packetStart = self.begin_KEXDH_REPLY()
    d = self.proto.ssh_KEX_DH_GEX_GROUP(packetStart + common.NS(b'bad signature'))
    return d.addCallback(lambda _: self.checkDisconnected(transport.DISCONNECT_KEY_EXCHANGE_FAILED))