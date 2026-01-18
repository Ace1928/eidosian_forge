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
def buildTLSProtocol(server=False, transport=None, fakeConnection=None, serverMethod=None):
    """
    Create a protocol hooked up to a TLS transport hooked up to a
    StringTransport.

    @param serverMethod: The TLS method accepted by the server-side and used by the created protocol. Set to to C{None} to use the default method used by your OpenSSL library.
    """
    clientProtocol = AccumulatingProtocol(999999999999)
    clientFactory = ClientFactory()
    clientFactory.protocol = lambda: clientProtocol
    if fakeConnection:

        @implementer(IOpenSSLServerConnectionCreator, IOpenSSLClientConnectionCreator)
        class HardCodedConnection:

            def clientConnectionForTLS(self, tlsProtocol):
                return fakeConnection
            serverConnectionForTLS = clientConnectionForTLS
        contextFactory = HardCodedConnection()
    elif server:
        contextFactory = ServerTLSContext(method=serverMethod)
    else:
        contextFactory = ClientTLSContext()
    clock = Clock()
    wrapperFactory = TLSMemoryBIOFactory(contextFactory, not server, clientFactory, clock)
    sslProtocol = wrapperFactory.buildProtocol(None)
    if transport is None:
        transport = StringTransport()
    sslProtocol.makeConnection(transport)
    return (clientProtocol, sslProtocol)