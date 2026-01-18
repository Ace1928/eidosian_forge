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
def test_clientConnectionLostWhileStopping(self):
    """
        When a client connection is lost while the service is stopping, the
        protocol stopping deferred is called and the reference to the protocol
        is removed.
        """
    clock = Clock()
    cq, service = self.makeReconnector(clock=clock)
    d = service.stopService()
    cq.constructedProtocols[0].connectionLost(Failure(IndentationError()))
    self.failureResultOf(service.whenConnected(), CancelledError)
    self.assertTrue(d.called)