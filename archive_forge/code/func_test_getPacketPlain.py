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
def test_getPacketPlain(self):
    """
        Test that packets are retrieved correctly out of the buffer when
        no encryption is enabled.
        """
    proto = MockTransportBase()
    proto.makeConnection(self.transport)
    self.finishKeyExchange(proto)
    self.transport.clear()
    proto.sendPacket(ord('A'), b'BC')
    proto.buf = self.transport.value() + b'extra'
    self.assertEqual(proto.getPacket(), b'ABC')
    self.assertEqual(proto.buf, b'extra')