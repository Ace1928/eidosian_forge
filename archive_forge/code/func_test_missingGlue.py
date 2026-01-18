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
def test_missingGlue(self):
    """
        If an intermediate response includes no glue records for the
        authorities, separate queries are made to find those addresses.
        """
    servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'authority': [(b'foo.example.com', Record_NS(b'ns1.example.org'))]}, (b'ns1.example.org', A): {'answers': [(b'ns1.example.org', Record_A('10.0.0.1'))]}}, ('10.0.0.1', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.2'))]}}}
    resolver = self._getResolver(servers)
    d = resolver.lookupAddress(b'foo.example.com')
    d.addCallback(getOneAddress)
    d.addCallback(self.assertEqual, '10.0.0.2')
    return d