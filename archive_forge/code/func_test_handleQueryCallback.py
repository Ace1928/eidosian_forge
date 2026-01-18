from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_handleQueryCallback(self):
    """
        L{server.DNSServerFactory.handleQuery} adds
        L{server.DNSServerFactory.resolver.gotResolverResponse} as a callback to
        the deferred returned by L{server.DNSServerFactory.resolver.query}. It
        is called with the query response, the original protocol, message and
        origin address.
        """
    f = server.DNSServerFactory()
    d = defer.Deferred()

    class FakeResolver:

        def query(self, *args, **kwargs):
            return d
    f.resolver = FakeResolver()
    gotResolverResponseArgs = []

    def fakeGotResolverResponse(*args, **kwargs):
        gotResolverResponseArgs.append((args, kwargs))
    f.gotResolverResponse = fakeGotResolverResponse
    m = dns.Message()
    m.addQuery(b'one.example.com')
    stubProtocol = NoopProtocol()
    dummyAddress = object()
    f.handleQuery(message=m, protocol=stubProtocol, address=dummyAddress)
    dummyResponse = object()
    d.callback(dummyResponse)
    self.assertEqual(gotResolverResponseArgs, [((dummyResponse, stubProtocol, m, dummyAddress), {})])