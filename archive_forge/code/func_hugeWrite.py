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
def hugeWrite(self, method=TLS_METHOD):
    """
        If a very long string is passed to L{TLSMemoryBIOProtocol.write}, any
        trailing part of it which cannot be send immediately is buffered and
        sent later.
        """
    data = b'some bytes'
    factor = 2 ** 20

    class SimpleSendingProtocol(Protocol):

        def connectionMade(self):
            self.transport.write(data * factor)
    clientFactory = ClientFactory()
    clientFactory.protocol = SimpleSendingProtocol
    clientContextFactory = HandshakeCallbackContextFactory(method=method)
    wrapperFactory = TLSMemoryBIOFactory(clientContextFactory, True, clientFactory)
    sslClientProtocol = wrapperFactory.buildProtocol(None)
    serverProtocol = AccumulatingProtocol(len(data) * factor)
    serverFactory = ServerFactory()
    serverFactory.protocol = lambda: serverProtocol
    serverContextFactory = ServerTLSContext(method=method)
    wrapperFactory = TLSMemoryBIOFactory(serverContextFactory, False, serverFactory)
    sslServerProtocol = wrapperFactory.buildProtocol(None)
    connectionDeferred = loopbackAsync(sslServerProtocol, sslClientProtocol)

    def cbConnectionDone(ignored):
        self.assertEqual(b''.join(serverProtocol.received), data * factor)
    connectionDeferred.addCallback(cbConnectionDone)
    return connectionDeferred