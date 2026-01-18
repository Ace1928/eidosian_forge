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
def test_bufferingContinuesIfProducerIsPausedOnWrite(self):
    """
        If the L{H2Connection} has buffered control frames, is unpaused, and then
        paused while unbuffering, it persists the buffer and stops trying to write.
        """

    class AutoPausingStringTransport(StringTransport):

        def write(self, *args, **kwargs):
            StringTransport.write(self, *args, **kwargs)
            self.producer.pauseProducing()
    connection = H2Connection()
    connection.requestFactory = DummyHTTPHandlerProxy
    frameFactory = FrameFactory()
    transport = AutoPausingStringTransport()
    transport.registerProducer(connection, True)
    clientConnectionPreface = frameFactory.clientConnectionPreface()
    connection.makeConnection(transport)
    connection.dataReceived(clientConnectionPreface)
    self.assertIsNotNone(connection._consumerBlocked)
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 1)
    self.assertEqual(connection._bufferedControlFrameBytes, 0)
    for _ in range(0, 11):
        connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 1)
    self.assertEqual(connection._bufferedControlFrameBytes, 9 * 11)
    connection.resumeProducing()
    frames = framesFromBytes(transport.value())
    self.assertEqual(len(frames), 2)
    self.assertEqual(connection._bufferedControlFrameBytes, 9 * 10)