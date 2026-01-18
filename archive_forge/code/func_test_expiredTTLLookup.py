import time
from zope.interface.verify import verifyClass
from twisted.internet import interfaces, task
from twisted.names import cache, dns
from twisted.trial import unittest
def test_expiredTTLLookup(self):
    """
        When the cache is queried exactly as the cached entry should expire but
        before it has actually been cleared, the cache does not return the
        expired entry.
        """
    r = ([dns.RRHeader(b'example.com', dns.A, dns.IN, 60, dns.Record_A('127.0.0.1', 60))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 50, dns.Record_A('127.0.0.1', 50))], [dns.RRHeader(b'example.com', dns.A, dns.IN, 40, dns.Record_A('127.0.0.1', 40))])
    clock = task.Clock()
    clock.callLater = lambda *args, **kwargs: None
    c = cache.CacheResolver({dns.Query(name=b'example.com', type=dns.A, cls=dns.IN): (clock.seconds(), r)}, reactor=clock)
    clock.advance(60.1)
    return self.assertFailure(c.lookupAddress(b'example.com'), dns.DomainError)