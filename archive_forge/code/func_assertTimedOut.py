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
def assertTimedOut(self, data, frameCount, errorCode, lastStreamID):
    """
        Confirm that the data that was sent matches what we expect from a
        timeout: namely, that it ends with a GOAWAY frame carrying an
        appropriate error code and last stream ID.
        """
    frames = framesFromBytes(data)
    self.assertEqual(len(frames), frameCount)
    self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
    self.assertEqual(frames[-1].error_code, errorCode)
    self.assertEqual(frames[-1].last_stream_id, lastStreamID)