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
def test_connectSSL(self):
    """
        L{ClientCreator.connectSSL} calls C{reactor.connectSSL} with the host,
        port, and context factory passed to it, and with a factory which will
        construct the protocol passed to L{ClientCreator.__init__}.
        """

    def check(reactor, cc):
        expectedContextFactory = object()
        cc.connectSSL('example.com', 1234, expectedContextFactory, 4321, ('4.3.2.1', 5678))
        host, port, factory, contextFactory, timeout, bindAddress = reactor.sslClients.pop()
        self.assertEqual(host, 'example.com')
        self.assertEqual(port, 1234)
        self.assertIs(contextFactory, expectedContextFactory)
        self.assertEqual(timeout, 4321)
        self.assertEqual(bindAddress, ('4.3.2.1', 5678))
        return factory
    self._basicConnectTest(check)