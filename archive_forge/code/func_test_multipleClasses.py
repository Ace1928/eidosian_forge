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
def test_multipleClasses(self):
    """
        Test that multiple instances have distinct states.
        """
    proto = self.proto
    proto.dataReceived(self.transport.value())
    proto.currentEncryptions = MockCipher()
    proto.outgoingCompression = MockCompression()
    proto.incomingCompression = MockCompression()
    proto.setService(MockService())
    proto2 = MockTransportBase()
    proto2.makeConnection(proto_helpers.StringTransport())
    proto2.sendIgnore(b'')
    self.assertNotEqual(proto.gotVersion, proto2.gotVersion)
    self.assertNotEqual(proto.transport, proto2.transport)
    self.assertNotEqual(proto.outgoingPacketSequence, proto2.outgoingPacketSequence)
    self.assertNotEqual(proto.incomingPacketSequence, proto2.incomingPacketSequence)
    self.assertNotEqual(proto.currentEncryptions, proto2.currentEncryptions)
    self.assertNotEqual(proto.service, proto2.service)