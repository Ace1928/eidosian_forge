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
def test_dataBeforeVersion(self):
    """
        Test that the transport ignores data sent before the version string.
        """
    proto = MockTransportBase()
    proto.makeConnection(proto_helpers.StringTransport())
    data = b"here's some stuff beforehand\nhere's some other stuff\n" + proto.ourVersionString + b'\r\n'
    [proto.dataReceived(c) for c in iterbytes(data)]
    self.assertTrue(proto.gotVersion)
    self.assertEqual(proto.otherVersionString, proto.ourVersionString)