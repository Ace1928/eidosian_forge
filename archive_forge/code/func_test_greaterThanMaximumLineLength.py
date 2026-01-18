import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_greaterThanMaximumLineLength(self):
    """
        C{LineOnlyReceiver} disconnects the transport if it receives a
        line longer than its C{MAX_LENGTH} + len(delimiter).
        """
    proto = LineOnlyTester()
    transport = proto_helpers.StringTransport()
    proto.makeConnection(transport)
    proto.dataReceived(b'x' * (proto.MAX_LENGTH + len(proto.delimiter) + 1) + b'\r\nr')
    self.assertTrue(transport.disconnecting)