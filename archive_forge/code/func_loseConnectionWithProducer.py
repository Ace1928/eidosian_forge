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
def loseConnectionWithProducer(self, writeBlockedOnRead):
    """
        Common code for tests involving writes by producer after
        loseConnection is called.
        """
    clientProtocol, tlsProtocol, producer = self.setupStreamingProducer()
    serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
    if not writeBlockedOnRead:
        self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
    else:
        pass
    clientProtocol.transport.write(b'x ')
    clientProtocol.transport.loseConnection()
    self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
    self.assertFalse(tlsProtocol.transport.disconnecting)
    self.assertFalse('stop' in producer.producerHistory)
    clientProtocol.transport.write(b'hello')
    clientProtocol.transport.writeSequence([b' ', b'world'])
    tlsProtocol.factory._clock.advance(0)
    clientProtocol.transport.unregisterProducer()
    self.assertNotEqual(tlsProtocol.transport.value(), b'')
    self.assertFalse(tlsProtocol.transport.disconnecting)
    clientProtocol.transport.write(b"won't")
    clientProtocol.transport.writeSequence([b"won't!"])
    tlsProtocol.factory._clock.advance(0)
    self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
    self.assertTrue(tlsProtocol.transport.disconnecting)
    self.assertEqual(b''.join(serverProtocol.received), b'x hello world')