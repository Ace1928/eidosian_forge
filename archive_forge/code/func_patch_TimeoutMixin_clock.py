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
def patch_TimeoutMixin_clock(self, connection, reactor):
    """
        Unfortunately, TimeoutMixin does not allow passing an explicit reactor
        to test timeouts. For that reason, we need to monkeypatch the method
        set up by the TimeoutMixin.

        @param connection: The HTTP/2 connection object to patch.
        @type connection: L{H2Connection}

        @param reactor: The reactor whose callLater method we want.
        @type reactor: An object implementing
            L{twisted.internet.interfaces.IReactorTime}
        """
    connection.callLater = reactor.callLater