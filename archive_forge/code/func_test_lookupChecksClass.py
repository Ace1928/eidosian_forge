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
def test_lookupChecksClass(self):
    """
        If a response includes a record with a class different from the one
        in the query, it is ignored and lookup continues until a record with
        the right class is found.
        """
    badClass = Record_A('10.0.0.1')
    badClass.CLASS = HS
    servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', badClass)], 'authority': [(b'foo.example.com', Record_NS(b'ns1.example.com'))], 'additional': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.3'))]}}}
    resolver = self._getResolver(servers)
    d = resolver.lookupAddress(b'foo.example.com')
    d.addCallback(getOnePayload)
    d.addCallback(self.assertEqual, Record_A('10.0.0.3'))
    return d