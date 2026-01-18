from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_delegationLookupEmpty(self):
    """
        If there are no records in the response to a lookup of a delegation
        nameserver, the L{Deferred} returned by L{Resolver.lookupAddress} fires
        with L{ResolverError}.
        """
    servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {}}}
    resolver = self._getResolver(servers)
    d = resolver.lookupAddress(b'example.com')
    return self.assertFailure(d, ResolverError)