from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleQueryErrback(self):
    """
        L{server.DNSServerFactory.handleQuery} adds
        L{server.DNSServerFactory.resolver.gotResolverError} as an errback to
        the deferred returned by L{server.DNSServerFactory.resolver.query}. It
        is called with the query failure, the original protocol, message and
        origin address.
        """
    f = server.DNSServerFactory()
    d = defer.Deferred()

    class FakeResolver:

        def query(self, *args, **kwargs):
            return d
    f.resolver = FakeResolver()
    gotResolverErrorArgs = []

    def fakeGotResolverError(*args, **kwargs):
        gotResolverErrorArgs.append((args, kwargs))
    f.gotResolverError = fakeGotResolverError
    m = dns.Message()
    m.addQuery(b'one.example.com')
    stubProtocol = NoopProtocol()
    dummyAddress = object()
    f.handleQuery(message=m, protocol=stubProtocol, address=dummyAddress)
    stubFailure = failure.Failure(Exception())
    d.errback(stubFailure)
    self.assertEqual(gotResolverErrorArgs, [((stubFailure, stubProtocol, m, dummyAddress), {})])