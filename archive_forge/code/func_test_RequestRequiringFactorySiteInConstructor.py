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
def test_RequestRequiringFactorySiteInConstructor(self):
    """
        A custom L{Request} subclass that requires the site and factory in the
        constructor is able to get them.
        """
    d = defer.Deferred()

    class SuperRequest(DummyHTTPHandler):

        def __init__(self, *args, **kwargs):
            DummyHTTPHandler.__init__(self, *args, **kwargs)
            d.callback((self.channel.site, self.channel.factory))
    connection = H2Connection()
    httpFactory = http.HTTPFactory()
    connection.requestFactory = _makeRequestProxyFactory(SuperRequest)
    connection.factory = httpFactory
    connection.site = object()
    self.connectAndReceive(connection, self.getRequestHeaders, [])

    def validateFactoryAndSite(args):
        site, factory = args
        self.assertIs(site, connection.site)
        self.assertIs(factory, connection.factory)
    d.addCallback(validateFactoryAndSite)
    cleanupCallback = connection._streamCleanupCallbacks[1]
    return defer.gatherResults([d, cleanupCallback])