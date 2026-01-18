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
def registerProducerAfterConnectionLost(self, streaming):
    """
        If a producer is registered after the transport has disconnected, the
        producer is not used, and its stopProducing method is called.
        """
    clientProtocol, tlsProtocol = buildTLSProtocol()
    clientProtocol.connectionLost = lambda reason: reason.trap(Error, ConnectionLost)

    class Producer:
        stopped = False

        def resumeProducing(self):
            return 1 / 0

        def stopProducing(self):
            self.stopped = True
    tlsProtocol.connectionLost(Failure(ConnectionDone()))
    producer = Producer()
    tlsProtocol.registerProducer(producer, False)
    self.assertIsNone(tlsProtocol.transport.producer)
    self.assertTrue(producer.stopped)