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
def test_initiallySchedulesOneDataCall(self):
    """
        When a H2Connection is established it schedules one call to be run as
        soon as the reactor has time.
        """
    reactor = task.Clock()
    a = H2Connection(reactor)
    calls = reactor.getDelayedCalls()
    self.assertEqual(len(calls), 1)
    call = calls[0]
    self.assertTrue(call.active())
    self.assertEqual(call.time, 0)
    self.assertEqual(call.func, a._sendPrioritisedData)
    self.assertEqual(call.args, ())
    self.assertEqual(call.kw, {})