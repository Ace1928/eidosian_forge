from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def test_disconnectAfterWriteAfterStartTLS(self):
    """
        L{ITCPTransport.loseConnection} ends a connection which was set up with
        L{ITLSTransport.startTLS} and which has recently been written to.  This
        is intended to verify that a socket send error masked by the TLS
        implementation doesn't prevent the connection from being reported as
        closed.
        """

    class ShortProtocol(Protocol):

        def connectionMade(self):
            if not ITLSTransport.providedBy(self.transport):
                finished = self.factory.finished
                self.factory.finished = None
                finished.errback(SkipTest('No ITLSTransport support'))
                return
            self.transport.startTLS(self.factory.context)
            self.transport.write(b'x')

        def dataReceived(self, data):
            self.transport.write(b'y')
            self.transport.loseConnection()

        def connectionLost(self, reason):
            finished = self.factory.finished
            if finished is not None:
                self.factory.finished = None
                finished.callback(reason)
    reactor = self.buildReactor()
    serverFactory = ServerFactory()
    serverFactory.finished = Deferred()
    serverFactory.protocol = ShortProtocol
    serverFactory.context = self.getServerContext()
    clientFactory = ClientFactory()
    clientFactory.finished = Deferred()
    clientFactory.protocol = ShortProtocol
    clientFactory.context = self.getClientContext()
    clientFactory.context.method = serverFactory.context.method
    lostConnectionResults = []
    finished = DeferredList([serverFactory.finished, clientFactory.finished], consumeErrors=True)

    def cbFinished(results):
        lostConnectionResults.extend([results[0][1], results[1][1]])
    finished.addCallback(cbFinished)
    port = reactor.listenTCP(0, serverFactory, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    connector = reactor.connectTCP(port.getHost().host, port.getHost().port, clientFactory)
    self.addCleanup(connector.disconnect)
    finished.addCallback(lambda ign: reactor.stop())
    self.runReactor(reactor)
    lostConnectionResults[0].trap(ConnectionClosed)
    lostConnectionResults[1].trap(ConnectionClosed)