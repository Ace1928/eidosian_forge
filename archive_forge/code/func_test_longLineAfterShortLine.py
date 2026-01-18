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
def test_longLineAfterShortLine(self):
    """
        If L{LineReceiver.dataReceived} is called with bytes representing a
        short line followed by bytes that exceed the length limit without a
        line delimiter, L{LineReceiver.lineLengthExceeded} is called with all
        of the bytes following the short line's delimiter.
        """
    excessive = b'x' * (self.proto.MAX_LENGTH * 2 + 2)
    self.proto.dataReceived(b'x' + self.proto.delimiter + excessive)
    self.assertEqual([excessive], self.proto.longLines)