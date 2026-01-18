import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_cancelTimeoutAlreadyCancelled(self):
    """
        Does nothing if the timeout is cancelled from another part.
        Ex from another thread.
        """
    sut, clock = self.getProtocolAndClock()
    sut.setTimeout(3)
    sut.timeoutCall.cancel()
    sut.cancelTimeout()
    self.assertFalse(sut.wrappedProtocol.disconnected)