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
def test_interleavedRequests(self):
    """
        Many interleaved POST requests all get received and responded to
        appropriately.
        """
    REQUEST_COUNT = 40
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyHTTPHandlerProxy
    streamIDs = list(range(1, REQUEST_COUNT * 2, 2))
    frames = [buildRequestFrames(self.postRequestHeaders, self.postRequestData, f, streamID) for streamID in streamIDs]
    requestBytes = f.clientConnectionPreface()
    frames = itertools.chain.from_iterable(zip(*frames))
    requestBytes += b''.join((frame.serialize() for frame in frames))
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)

    def validate(results):
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 1 + 3 * 40)
        for streamID in streamIDs:
            streamFrames = [f for f in frames if f.stream_id == streamID and (not isinstance(f, hyperframe.frame.WindowUpdateFrame))]
            self.assertEqual(len(streamFrames), 3)
            self.assertEqual(dict(streamFrames[0].data), dict(self.postResponseHeaders))
            self.assertEqual(streamFrames[1].data, self.postResponseData)
            self.assertEqual(streamFrames[2].data, b'')
            self.assertTrue('END_STREAM' in streamFrames[2].flags)
    return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)