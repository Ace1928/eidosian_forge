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
def test_clearLineBuffer(self):
    """
        L{LineReceiver.clearLineBuffer} removes all buffered data and returns
        it as a C{bytes} and can be called from beneath C{dataReceived}.
        """

    class ClearingReceiver(basic.LineReceiver):

        def lineReceived(self, line):
            self.line = line
            self.rest = self.clearLineBuffer()
    protocol = ClearingReceiver()
    protocol.dataReceived(b'foo\r\nbar\r\nbaz')
    self.assertEqual(protocol.line, b'foo')
    self.assertEqual(protocol.rest, b'bar\r\nbaz')
    protocol.dataReceived(b'quux\r\n')
    self.assertEqual(protocol.line, b'quux')
    self.assertEqual(protocol.rest, b'')