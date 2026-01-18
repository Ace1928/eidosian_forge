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
def test_followCanonicalName(self):
    """
        If no record of the requested type is included in a response, but a
        I{CNAME} record for the query name is included, queries are made to
        resolve the value of the I{CNAME}.
        """
    servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net'))]}, (b'example.net', A): {'answers': [(b'example.net', Record_A('10.0.0.5'))]}}}
    resolver = self._getResolver(servers)
    d = resolver.lookupAddress(b'example.com')
    d.addCallback(lambda results: results[0])
    d.addCallback(self.assertEqual, [RRHeader(b'example.com', CNAME, payload=Record_CNAME(b'example.net')), RRHeader(b'example.net', A, payload=Record_A('10.0.0.5'))])
    return d