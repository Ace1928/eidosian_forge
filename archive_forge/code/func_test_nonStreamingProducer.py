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
def test_nonStreamingProducer(self):
    """
        Non-streaming producers get wrapped as streaming producers.
        """
    clientProtocol, tlsProtocol = buildTLSProtocol()
    producer = NonStreamingProducer(clientProtocol.transport)
    clientProtocol.transport.registerProducer(producer, False)
    streamingProducer = tlsProtocol.transport.producer._producer
    self.assertIsInstance(streamingProducer, _PullToPush)
    self.assertEqual(streamingProducer._producer, producer)
    self.assertEqual(streamingProducer._consumer, clientProtocol.transport)
    self.assertTrue(tlsProtocol.transport.streaming)

    def done(ignore):
        self.assertIsNone(producer.consumer)
        self.assertIsNone(tlsProtocol.transport.producer)
        self.assertTrue(streamingProducer._finished)
    producer.result.addCallback(done)
    serverProtocol, serverTLSProtocol = buildTLSProtocol(server=True)
    self.flushTwoTLSProtocols(tlsProtocol, serverTLSProtocol)
    return producer.result