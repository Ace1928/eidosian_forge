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
def buildGoAwayFrame(self, lastStreamID, errorCode=0, additionalData=b''):
    """
        Builds a single GOAWAY frame.
        """
    f = hyperframe.frame.GoAwayFrame(0)
    f.error_code = errorCode
    f.last_stream_id = lastStreamID
    f.additional_data = additionalData
    return f