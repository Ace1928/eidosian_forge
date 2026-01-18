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
def test_cancelConnectTCPFailedTimeout(self):
    """
        Similar to L{test_cancelConnectTCPTimeout}, but for the case where the
        connection attempt fails.
        """

    def connect(reactor, cc):
        d = cc.connectTCP('example.com', 1234)
        host, port, factory, timeout, bindAddress = reactor.tcpClients.pop()
        return (d, factory)
    return self._cancelConnectFailedTimeoutTest(connect)