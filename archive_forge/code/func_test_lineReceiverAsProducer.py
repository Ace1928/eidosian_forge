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
def test_lineReceiverAsProducer(self):
    """
        Test produce/unproduce in receiving.
        """
    a = LineTester()
    t = proto_helpers.StringIOWithoutClosing()
    a.makeConnection(protocol.FileWrapper(t))
    a.dataReceived(b'produce\nhello world\nunproduce\ngoodbye\n')
    self.assertEqual(a.received, [b'produce', b'hello world', b'unproduce', b'goodbye'])