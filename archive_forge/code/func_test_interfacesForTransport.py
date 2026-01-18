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
def test_interfacesForTransport(self):
    """
        If the protocol objects returned by the factory given to
        L{ClientService} provide special "marker" interfaces for their
        transport - L{IHalfCloseableProtocol} or L{IFileDescriptorReceiver} -
        those interfaces will be provided by the protocol objects passed on to
        the reactor.
        """

    @implementer(IHalfCloseableProtocol, IFileDescriptorReceiver)
    class FancyProtocol(Protocol):
        """
            Provider of various interfaces.
            """
    cq, service = self.makeReconnector(protocolType=FancyProtocol)
    reactorFacing = cq.constructedProtocols[0]
    self.assertTrue(IFileDescriptorReceiver.providedBy(reactorFacing))
    self.assertTrue(IHalfCloseableProtocol.providedBy(reactorFacing))