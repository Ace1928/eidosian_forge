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
def test_getPacketEncrypted(self):
    """
        Test that encrypted packets are retrieved correctly.
        See test_sendPacketEncrypted.
        """
    proto = MockTransportBase()
    proto.sendKexInit = lambda: None
    proto.makeConnection(self.transport)
    self.transport.clear()
    proto.currentEncryptions = testCipher = MockCipher()
    proto.sendPacket(ord('A'), b'BCD')
    value = self.transport.value()
    proto.buf = value[:MockCipher.decBlockSize]
    self.assertIsNone(proto.getPacket())
    self.assertTrue(testCipher.usedDecrypt)
    self.assertEqual(proto.first, b'\x00\x00\x00\x0e\tA')
    proto.buf += value[MockCipher.decBlockSize:]
    self.assertEqual(proto.getPacket(), b'ABCD')
    self.assertEqual(proto.buf, b'')