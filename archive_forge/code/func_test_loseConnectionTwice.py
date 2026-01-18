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
def test_loseConnectionTwice(self):
    """
        If TLSMemoryBIOProtocol.loseConnection is called multiple times, all
        but the first call have no effect.
        """
    tlsClient, tlsServer, handshakeDeferred, disconnectDeferred = self.handshakeProtocols()
    self.successResultOf(handshakeDeferred)
    calls = []

    def _shutdownTLS(shutdown=tlsClient._shutdownTLS):
        calls.append(1)
        return shutdown()
    tlsClient._shutdownTLS = _shutdownTLS
    tlsClient.write(b'x')
    tlsClient.loseConnection()
    self.assertTrue(tlsClient.disconnecting)
    self.assertEqual(calls, [1])
    tlsClient.loseConnection()
    self.assertEqual(calls, [1])
    return disconnectDeferred