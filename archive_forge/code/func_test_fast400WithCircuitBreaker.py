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
def test_fast400WithCircuitBreaker(self):
    """
        A HTTP/2 stream that has had _respondToBadRequestAndDisconnect
        called on it does not write control frame data if its
        transport is paused and its control frame limit has been
        reached.
        """
    memoryReactor = MemoryReactorClock()
    connection = H2Connection(memoryReactor)
    connection.callLater = memoryReactor.callLater
    connection.requestFactory = DelayedHTTPHandler
    streamID = 1
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    connection.dataReceived(buildRequestBytes(self.getRequestHeaders, [], frameFactory, streamID=streamID))
    connection.pauseProducing()
    connection._maxBufferedControlFrameBytes = 0
    connection._respondToBadRequestAndDisconnect(streamID)
    self.assertTrue(transport.disconnected)