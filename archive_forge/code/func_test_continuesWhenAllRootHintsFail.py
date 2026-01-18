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
def test_continuesWhenAllRootHintsFail(self):
    """
        The L{root.Resolver} is eventually created, even if all of the root hint
        lookups fail. Pending and new lookups will then fail with
        AttributeError.
        """
    stubResolver = StubResolver()
    deferredResolver = root.bootstrap(stubResolver)
    results = iter(stubResolver.pendingResults)
    d1 = next(results)
    for d in results:
        d.errback(TimeoutError())
    d1.errback(TimeoutError())

    def checkHints(res):
        self.assertEqual(deferredResolver.hints, [])
    d1.addBoth(checkHints)
    self.addCleanup(self.flushLoggedErrors, TimeoutError)