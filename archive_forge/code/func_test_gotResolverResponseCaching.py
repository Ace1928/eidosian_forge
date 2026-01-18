from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def test_gotResolverResponseCaching(self):
    """
        L{server.DNSServerFactory.gotResolverResponse} caches the response if at
        least one cache was provided in the constructor.
        """
    f = NoResponseDNSServerFactory(caches=[RaisingCache()])
    m = dns.Message()
    m.addQuery(b'example.com')
    expectedAnswers = [dns.RRHeader()]
    expectedAuthority = []
    expectedAdditional = []
    e = self.assertRaises(RaisingCache.CacheResultArguments, f.gotResolverResponse, (expectedAnswers, expectedAuthority, expectedAdditional), protocol=NoopProtocol(), message=m, address=None)
    (query, (answers, authority, additional)), kwargs = e.args
    self.assertEqual(query.name.name, b'example.com')
    self.assertIs(answers, expectedAnswers)
    self.assertIs(authority, expectedAuthority)
    self.assertIs(additional, expectedAdditional)