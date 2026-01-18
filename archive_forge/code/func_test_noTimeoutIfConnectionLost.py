import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_noTimeoutIfConnectionLost(self):
    """
        When a L{H2Connection} loses its connection it cancels its timeout.
        """
    frameFactory = FrameFactory()
    frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
    initialData = frameFactory.clientConnectionPreface()
    initialData += b''.join((f.serialize() for f in frames))
    reactor, conn, transport = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
    sentData = transport.value()
    oldCallCount = len(reactor.getDelayedCalls())
    conn.connectionLost('reason')
    currentCallCount = len(reactor.getDelayedCalls())
    self.assertEqual(oldCallCount - 1, currentCallCount)
    reactor.advance(101)
    self.assertEqual(transport.value(), sentData)