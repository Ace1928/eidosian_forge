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
def test_filteredQuery(self):
    """
        L{Resolver._query} accepts a L{Query} instance and an address, issues
        the query, and returns a L{Deferred} which fires with the response to
        the query.  If a true value is passed for the C{filter} parameter, the
        result is a three-tuple of lists of records.
        """
    answer, authority, additional = self._queryTest(True)
    self.assertEqual(answer, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
    self.assertEqual(authority, [])
    self.assertEqual(additional, [])