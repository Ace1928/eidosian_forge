import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_firstBoxFirstKeyExcessiveLength(self):
    """
        L{amp.BinaryBoxProtocol} drops its connection if the length prefix for
        the first a key it receives is larger than 255.
        """
    transport = StringTransport()
    protocol = amp.BinaryBoxProtocol(self)
    protocol.makeConnection(transport)
    protocol.dataReceived(b'\x01\x00')
    self.assertTrue(transport.disconnecting)