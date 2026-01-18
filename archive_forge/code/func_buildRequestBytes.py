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
def buildRequestBytes(headers, data, frameFactory=None, streamID=1):
    """
    Provides the byte sequence for a collection of HTTP/2 frames representing
    the provided request.

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
    frames = buildRequestFrames(headers, data, frameFactory, streamID)
    return b''.join((f.serialize() for f in frames))