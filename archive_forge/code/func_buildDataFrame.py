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
def buildDataFrame(self, data, flags=None, streamID=1):
    """
        Builds a single data frame out of a chunk of data.
        """
    flags = set(flags) if flags is not None else set()
    f = hyperframe.frame.DataFrame(streamID)
    f.data = data
    f.flags = flags
    return f