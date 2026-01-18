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
def test_failWithProtocolError(self):
    """
        A HTTP/2 protocol error triggers the L{http.Request.notifyFinish}
        deferred for all outstanding requests with a Failure that contains the
        underlying exception.
        """
    connection = H2Connection()
    connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
    frameFactory, transport = self.connectAndReceive(connection, self.getRequestHeaders, [])
    secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
    connection.dataReceived(secondRequest)
    deferreds = connection.requestFactory.results
    self.assertEqual(len(deferreds), 2)

    def callback(ign):
        self.fail("Didn't errback, called back instead")

    def errback(reason):
        self.assertIsInstance(reason, failure.Failure)
        self.assertIsInstance(reason.value, h2.exceptions.ProtocolError)
        return None
    for d in deferreds:
        d.addCallbacks(callback, errback)
    invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
    connection.dataReceived(invalidData)
    return defer.gatherResults(deferreds)