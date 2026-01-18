from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_caches(self):
    """
        L{server.DNSServerFactory.__init__} accepts a C{caches} argument. The
        value of this argument is a list and is used to extend the C{resolver}
        L{resolve.ResolverChain}.
        """
    dummyResolver = object()
    self.assertEqual(server.DNSServerFactory(caches=[dummyResolver]).resolver.resolvers, [dummyResolver])