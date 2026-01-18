from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_loseConnectionAfterHandshake(self):
    """
        L{TLSMemoryBIOProtocol.loseConnection} sends a TLS close alert and
        shuts down the underlying connection cleanly on both sides, after
        transmitting all buffered data.
        """

    class NotifyingProtocol(ConnectionLostNotifyingProtocol):

        def __init__(self, onConnectionLost):
            ConnectionLostNotifyingProtocol.__init__(self, onConnectionLost)
            self.data = []

        def dataReceived(self, data):
            self.data.append(data)
    clientConnectionLost = Deferred()
    clientFactory = ClientFactory()
    clientProtocol = NotifyingProtocol(clientConnectionLost)
    clientFactory.protocol = lambda: clientProtocol
    clientContextFactory, handshakeDeferred = HandshakeCallbackContextFactory.factoryAndDeferred()
    wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
    sslClientProtocol = wrapperFactory.buildProtocol(None)
    serverConnectionLost = Deferred()
    serverProtocol = NotifyingProtocol(serverConnectionLost)
    serverFactory = ServerFactory()
    serverFactory.protocol = lambda: serverProtocol
    serverContextFactory = ServerTLSContext()
    wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
    sslServerProtocol = wrapperFactory.buildProtocol(None)
    loopbackAsync(sslServerProtocol, sslClientProtocol)
    chunkOfBytes = b'123456890' * 100000

    def cbHandshake(ignored):
        clientProtocol.transport.write(chunkOfBytes)
        serverProtocol.transport.write(b'x')
        serverProtocol.transport.loseConnection()
        return gatherResults([clientConnectionLost, serverConnectionLost])
    handshakeDeferred.addCallback(cbHandshake)

    def cbConnectionDone(result):
        clientProtocol, serverProtocol = result
        clientProtocol.lostConnectionReason.trap(ConnectionDone)
        serverProtocol.lostConnectionReason.trap(ConnectionDone)
        self.assertEqual(b''.join(serverProtocol.data), chunkOfBytes)
        self.assertTrue(serverProtocol.transport.q.disconnect)
        self.assertTrue(clientProtocol.transport.q.disconnect)
    handshakeDeferred.addCallback(cbConnectionDone)
    return handshakeDeferred