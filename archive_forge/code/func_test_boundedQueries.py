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
def test_boundedQueries(self):
    """
        L{Resolver.lookupAddress} won't issue more queries following
        delegations than the limit passed to its initializer.
        """
    servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {'answers': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns2.example.com'))], 'additional': [(b'ns2.example.com', Record_A('10.0.0.3'))]}}, ('10.0.0.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_A('10.0.0.4'))]}}}
    failer = self._getResolver(servers, 3)
    failD = self.assertFailure(failer.lookupAddress(b'example.com'), ResolverError)
    succeeder = self._getResolver(servers, 4)
    succeedD = succeeder.lookupAddress(b'example.com')
    succeedD.addCallback(getOnePayload)
    succeedD.addCallback(self.assertEqual, Record_A('10.0.0.4'))
    return gatherResults([failD, succeedD])