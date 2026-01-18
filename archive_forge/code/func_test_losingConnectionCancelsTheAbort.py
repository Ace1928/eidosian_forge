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
def test_losingConnectionCancelsTheAbort(self):
    """
        When a L{H2Connection} has timed the connection out, getting
        C{connectionLost} called on it cancels the forcible connection close.
        """
    reactor, conn, transport = self.prepareAbortTest()
    reactor.advance(14)
    conn.connectionLost(None)
    reactor.advance(1)
    self.assertTrue(transport.disconnecting)
    self.assertFalse(transport.disconnected)