from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_cancelConnectTCPTimeout(self):
    """
        L{ClientCreator.connectTCP} inserts a very short delayed call between
        the time the connection is established and the time the L{Deferred}
        returned from one of its connect methods actually fires.  If the
        L{Deferred} is cancelled in this interval, the established connection is
        closed, the timeout is cancelled, and the L{Deferred} fails with
        L{CancelledError}.
        """

    def connect(reactor, cc):
        d = cc.connectTCP('example.com', 1234)
        host, port, factory, timeout, bindAddress = reactor.tcpClients.pop()
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        return d
    return self._cancelConnectTimeoutTest(connect)