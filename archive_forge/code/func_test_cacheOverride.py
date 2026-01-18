from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_cacheOverride(self):
    """
        L{server.DNSServerFactory.__init__} assigns the last object in the
        C{caches} list to L{server.DNSServerFactory.cache}.
        """
    dummyResolver = object()
    self.assertEqual(server.DNSServerFactory(caches=[object(), dummyResolver]).cache, dummyResolver)