import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_cancelTimeout(self):
    """
        Setting the timeout to L{None} cancel any timeout operations.
        """
    self.proto.timeOut = 5
    self.proto.makeConnection(StringTransport())
    self.proto.setTimeout(None)
    self.assertIsNone(self.proto.timeOut)
    self.clock.pump([0, 5, 5, 5])
    self.assertFalse(self.proto.timedOut)