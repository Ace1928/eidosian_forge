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
def test_bufferingAutomaticFrameData(self):
    """
        If a the L{H2Connection} has been paused by the transport, it will
        not write automatic frame data triggered by writes.
        """
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    frameFactory = FrameFactory()
    transport = StringTransport()
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    connection.pauseProducing()
    for _ in range(0, 100):
        connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 1)
    connection.resumeProducing()
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 101)