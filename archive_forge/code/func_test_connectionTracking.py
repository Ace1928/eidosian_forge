import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
def test_connectionTracking(self):
    """
        L{dns.DNSProtocol} calls its controller's C{connectionMade}
        method with itself when it is connected to a transport and its
        controller's C{connectionLost} method when it is disconnected.
        """
    self.assertEqual(self.controller.connections, [self.proto])
    self.proto.connectionLost(Failure(ConnectionDone('Fake Connection Done')))
    self.assertEqual(self.controller.connections, [])