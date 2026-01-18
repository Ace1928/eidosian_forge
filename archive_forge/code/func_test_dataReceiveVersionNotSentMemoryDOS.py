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
def test_dataReceiveVersionNotSentMemoryDOS(self):
    """
        When the peer is not sending its SSH version but keeps sending data,
        the connection is disconnected after 4KB to prevent buffering too
        much and running our of memory.
        """
    sut = MockTransportBase()
    sut.makeConnection(self.transport)
    sut.dataReceived(b'SSH-2-Server-Identifier')
    sut.dataReceived(b'1234567890' * 406)
    sut.dataReceived(b'1235678')
    self.assertFalse(self.transport.disconnecting)
    sut.dataReceived(b'1234567')
    self.assertTrue(self.transport.disconnecting)
    self.assertIn(b'Preventing a denial of service attack', self.transport.value())