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
def connectAndReceive(self, connection, headers, body):
    """
        Takes a single L{H2Connection} object and connects it to a
        L{StringTransport} using a brand new L{FrameFactory}.

        @param connection: The L{H2Connection} object to connect.
        @type connection: L{H2Connection}

        @param headers: The headers to send on the first request.
        @type headers: L{Iterable} of L{tuple} of C{(bytes, bytes)}

        @param body: Chunks of body to send, if any.
        @type body: L{Iterable} of L{bytes}

        @return: A tuple of L{FrameFactory}, L{StringTransport}
        """
    frameFactory = FrameFactory()
    transport = StringTransport()
    requestBytes = frameFactory.clientConnectionPreface()
    requestBytes += buildRequestBytes(headers, body, frameFactory)
    connection.makeConnection(transport)
    for byte in iterbytes(requestBytes):
        connection.dataReceived(byte)
    return (frameFactory, transport)