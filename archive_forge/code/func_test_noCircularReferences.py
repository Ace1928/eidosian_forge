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
def test_noCircularReferences(self):
    """
        TLSMemoryBIOProtocol doesn't leave circular references that keep
        it in memory after connection is closed.
        """

    def nObjectsOfType(type):
        """
            Return the number of instances of a given type in memory.

            @param type: Type whose instances to find.

            @return: The number of instances found.
            """
        return sum((1 for x in gc.get_objects() if isinstance(x, type)))
    self.addCleanup(gc.enable)
    gc.disable()

    class CloserProtocol(Protocol):

        def dataReceived(self, data):
            self.transport.loseConnection()

    class GreeterProtocol(Protocol):

        def connectionMade(self):
            self.transport.write(b'hello')
    origTLSProtos = nObjectsOfType(TLSMemoryBIOProtocol)
    origServerProtos = nObjectsOfType(CloserProtocol)
    authCert, serverCert = certificatesForAuthorityAndServer()
    serverFactory = TLSMemoryBIOFactory(serverCert.options(), False, Factory.forProtocol(CloserProtocol))
    clientFactory = TLSMemoryBIOFactory(optionsForClientTLS('example.com', trustRoot=authCert), True, Factory.forProtocol(GreeterProtocol))
    loopbackAsync(TLSMemoryBIOProtocol(serverFactory, CloserProtocol()), TLSMemoryBIOProtocol(clientFactory, GreeterProtocol()))
    newTLSProtos = nObjectsOfType(TLSMemoryBIOProtocol)
    newServerProtos = nObjectsOfType(CloserProtocol)
    self.assertEqual(newTLSProtos, origTLSProtos)
    self.assertEqual(newServerProtos, origServerProtos)