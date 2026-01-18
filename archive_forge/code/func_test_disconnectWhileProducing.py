import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
def test_disconnectWhileProducing(self):
    """
        If C{loseConnection} is called while a producer is registered with the
        transport, the connection is closed after the producer is unregistered.
        """
    reactor = self.buildReactor()
    skippedReactors = ['Glib2Reactor', 'Gtk2Reactor']
    reactorClassName = reactor.__class__.__name__
    if reactorClassName in skippedReactors and platform.isWindows():
        raise SkipTest('A pygobject/pygtk bug disables this functionality on Windows.')

    class Producer:

        def resumeProducing(self):
            log.msg('Producer.resumeProducing')
    self.listen(reactor, ServerFactory.forProtocol(Protocol))
    finished = Deferred()
    finished.addErrback(log.err)
    finished.addCallback(lambda ign: reactor.stop())

    class ClientProtocol(Protocol):
        """
            Protocol to connect, register a producer, try to lose the
            connection, unregister the producer, and wait for the connection to
            actually be lost.
            """

        def connectionMade(self):
            log.msg('ClientProtocol.connectionMade')
            self.transport.registerProducer(Producer(), False)
            self.transport.loseConnection()
            reactor.callLater(0, reactor.callLater, 0, self.unregister)

        def unregister(self):
            log.msg('ClientProtocol unregister')
            self.transport.unregisterProducer()
            reactor.callLater(1.0, finished.errback, Failure(Exception('Connection was not lost')))

        def connectionLost(self, reason):
            log.msg('ClientProtocol.connectionLost')
            finished.callback(None)
    clientFactory = ClientFactory()
    clientFactory.protocol = ClientProtocol
    self.connect(reactor, clientFactory)
    self.runReactor(reactor)