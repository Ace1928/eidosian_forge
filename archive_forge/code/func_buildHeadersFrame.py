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
def buildHeadersFrame(self, headers, flags=[], streamID=1, **priorityKwargs):
    """
        Builds a single valid headers frame out of the contained headers.
        """
    f = hyperframe.frame.HeadersFrame(streamID)
    f.data = self.encoder.encode(headers)
    f.flags.add('END_HEADERS')
    for flag in flags:
        f.flags.add(flag)
    for k, v in priorityKwargs.items():
        setattr(f, k, v)
    return f