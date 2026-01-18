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
def test_maximumLineLengthPartialDelimiter(self):
    """
        C{LineReceiver} doesn't disconnect the transport when it
        receives a finished line as long as its C{MAX_LENGTH}, when
        the second-to-last packet ended with a pattern that could have
        been -- and turns out to have been -- the start of a
        delimiter, and that packet causes the total input to exceed
        C{MAX_LENGTH} + len(delimiter).
        """
    proto = LineTester()
    proto.MAX_LENGTH = 4
    t = proto_helpers.StringTransport()
    proto.makeConnection(t)
    line = b'x' * (proto.MAX_LENGTH - 1)
    proto.dataReceived(line)
    proto.dataReceived(proto.delimiter[:-1])
    proto.dataReceived(proto.delimiter[-1:] + line)
    self.assertFalse(t.disconnecting)
    self.assertEqual(len(proto.received), 1)
    self.assertEqual(line, proto.received[0])