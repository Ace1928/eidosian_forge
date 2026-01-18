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
def endpointForTesting(fireImmediately=False):
    """
    Make a sample endpoint for testing.

    @param fireImmediately: If true, fire all L{Deferred}s returned from
        C{connect} immedaitely.

    @return: a 2-tuple of C{(information, endpoint)}, where C{information} is a
        L{ConnectInformation} describing the operations in progress on
        C{endpoint}.
    """

    @implementer(IStreamClientEndpoint)
    class ClientTestEndpoint:

        def connect(self, factory):
            result = Deferred()
            info.passedFactories.append(factory)

            @result.addCallback
            def createProtocol(ignored):
                protocol = factory.buildProtocol(None)
                info.constructedProtocols.append(protocol)
                transport = StringTransport()
                protocol.makeConnection(transport)
                return protocol
            info.connectQueue.append(result)
            if fireImmediately:
                result.callback(None)
            return result
    info = ConnectInformation()
    return (info, ClientTestEndpoint())