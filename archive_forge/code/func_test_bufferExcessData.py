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
def test_bufferExcessData(self):
    """
        When a L{Request} object is not using C{IProducer} to generate data and
        so is not having backpressure exerted on it, the L{H2Stream} object
        will buffer data until the flow control window is opened.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    requestBytes = f.clientConnectionPreface()
    requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
    requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    bonusFrames = len(self.getResponseData) - 5
    for _ in range(bonusFrames):
        frame = f.buildWindowUpdateFrame(streamID=1, increment=1)
        a.dataReceived(frame.serialize())

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertTrue('END_STREAM' in frames[-1].flags)
        actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
        self.assertEqual(self.getResponseData, actualResponseData)
    return a._streamCleanupCallbacks[1].addCallback(validate)