import time
from zope.interface.verify import verifyClass
from twisted.internet import interfaces, task
from twisted.names import cache, dns
from twisted.trial import unittest
def test_cachedResultExpires(self):
    """
        Once the TTL has been exceeded, the result is removed from the cache.
        """
    r = ([dns.RRHeader(b'example.com', dns.A, dns.IN, 60, dns.Record_A('127.0.0.1', 60))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 50, dns.Record_A('127.0.0.1', 50))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 40, dns.Record_A('127.0.0.1', 40))])
    clock = task.Clock()
    c = cache.CacheResolver(reactor=clock)
    query = dns.Query(name=b'example.com', type=dns.A, cls=dns.IN)
    c.cacheResult(query, r)
    clock.advance(40)
    self.assertNotIn(query, c.cache)
    return self.assertFailure(c.lookupAddress(b'example.com'), dns.DomainError)