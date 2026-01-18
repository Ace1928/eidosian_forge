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
def test_connectUNIX(self):
    """
        L{ClientCreator.connectUNIX} calls C{reactor.connectUNIX} with the
        filename passed to it, and with a factory which will construct the
        protocol passed to L{ClientCreator.__init__}.
        """

    def check(reactor, cc):
        cc.connectUNIX('/foo/bar', 123, True)
        address, factory, timeout, checkPID = reactor.unixClients.pop()
        self.assertEqual(address, '/foo/bar')
        self.assertEqual(timeout, 123)
        self.assertTrue(checkPID)
        return factory
    self._basicConnectTest(check)