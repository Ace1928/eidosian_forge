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
def buildRequestFrames(headers, data, frameFactory=None, streamID=1):
    """
    Provides a sequence of HTTP/2 frames that encode a single HTTP request.
    This should be used when you want to control the serialization yourself,
    e.g. because you want to interleave other frames with these. If that's not
    necessary, prefer L{buildRequestBytes}.

    @param headers: The HTTP/2 headers to send.
    @type headers: L{list} of L{tuple} of L{bytes}

    @param data: The HTTP data to send. Each list entry will be sent in its own
    frame.
    @type data: L{list} of L{bytes}

    @param frameFactory: The L{FrameFactory} that will be used to construct the
    frames.
    @type frameFactory: L{FrameFactory}

    @param streamID: The ID of the stream on which to send the request.
    @type streamID: L{int}
    """
    if frameFactory is None:
        frameFactory = FrameFactory()
    frames = []
    frames.append(frameFactory.buildHeadersFrame(headers=headers, streamID=streamID))
    frames.extend((frameFactory.buildDataFrame(chunk, streamID=streamID) for chunk in data))
    frames[-1].flags.add('END_STREAM')
    return frames