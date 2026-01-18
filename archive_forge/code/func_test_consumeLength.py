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
def test_consumeLength(self):
    """
        C{_consumeLength} returns the expected length of the
        netstring, including the trailing comma.
        """
    self.netstringReceiver._remainingData = b'12:'
    self.netstringReceiver._consumeLength()
    self.assertEqual(self.netstringReceiver._expectedPayloadSize, 13)