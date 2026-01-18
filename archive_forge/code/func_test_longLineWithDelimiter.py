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
def test_longLineWithDelimiter(self):
    """
        If L{LineReceiver.dataReceived} is called with more than
        C{LineReceiver.MAX_LENGTH} bytes containing a line delimiter somewhere
        not in the first C{MAX_LENGTH} bytes, the entire byte string is passed
        to L{LineReceiver.lineLengthExceeded}.
        """
    excessive = self.proto.delimiter.join([b'x' * (self.proto.MAX_LENGTH * 2 + 2)] * 2)
    self.proto.dataReceived(excessive)
    self.assertEqual([excessive], self.proto.longLines)