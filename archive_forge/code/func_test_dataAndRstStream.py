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
def test_dataAndRstStream(self):
    """
        When a DATA frame is received at the same time as RST_STREAM,
        Twisted does not send WINDOW_UPDATE frames for the stream.
        """
    frameFactory = FrameFactory()
    transport = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    frameData = [b'\x00' * 2 ** 14] * 4
    bodyLength = f'{sum((len(data) for data in frameData))}'
    headers = self.postRequestHeaders[:-1] + [('content-length', bodyLength)]
    frames = buildRequestFrames(headers=headers, data=frameData, frameFactory=frameFactory)
    del frames[-1]
    frames.append(frameFactory.buildRstStreamFrame(streamID=1, errorCode=h2.errors.ErrorCodes.INTERNAL_ERROR))
    requestBytes = frameFactory.clientConnectionPreface()
    requestBytes += b''.join((f.serialize() for f in frames))
    a.makeConnection(transport)
    a.dataReceived(requestBytes)
    frames = framesFromBytes(transport.value())
    windowUpdateFrameIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.WindowUpdateFrame)]
    self.assertEqual([0], windowUpdateFrameIDs)
    headersFrames = [f for f in frames if isinstance(f, hyperframe.frame.HeadersFrame)]
    dataFrames = [f for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
    self.assertFalse(headersFrames)
    self.assertFalse(dataFrames)