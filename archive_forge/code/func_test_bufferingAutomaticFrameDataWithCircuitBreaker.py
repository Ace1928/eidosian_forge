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
def test_bufferingAutomaticFrameDataWithCircuitBreaker(self):
    """
        If the L{H2Connection} has been paused by the transport, it will
        not write automatic frame data triggered by writes. If this buffer
        gets too large, the connection will be dropped.
        """
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    connection.pauseProducing()
    connection._maxBufferedControlFrameBytes = 100
    self.assertFalse(transport.disconnecting)
    for _ in range(0, 11):
        connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
    self.assertFalse(transport.disconnecting)
    connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
    self.assertTrue(transport.disconnected)