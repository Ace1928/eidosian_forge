import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_clientConnected(self):
    """
        When a client connects, the service keeps a reference to the new
        protocol and resets the delay.
        """
    clock = Clock()
    cq, service = self.makeReconnector(clock=clock)
    awaitingProtocol = service.whenConnected()
    self.assertEqual(clock.getDelayedCalls(), [])
    self.assertIdentical(self.successResultOf(awaitingProtocol), cq.applicationProtocols[0])