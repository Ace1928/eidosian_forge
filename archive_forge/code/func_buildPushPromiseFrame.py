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
def buildPushPromiseFrame(self, streamID, promisedStreamID, headers, flags=[]):
    """
        Builds a single Push Promise frame.
        """
    f = hyperframe.frame.PushPromiseFrame(streamID)
    f.promised_stream_id = promisedStreamID
    f.data = self.encoder.encode(headers)
    f.flags = set(flags)
    f.flags.add('END_HEADERS')
    return f