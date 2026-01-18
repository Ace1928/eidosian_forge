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
def test_continuesWhenSomeRootHintsFail(self):
    """
        The L{root.Resolver} is eventually created, even if some of the root
        hint lookups fail. Only the working root hint IP addresses are supplied
        to the L{root.Resolver}.
        """
    stubResolver = StubResolver()
    deferredResolver = root.bootstrap(stubResolver)
    results = iter(stubResolver.pendingResults)
    d1 = next(results)
    for d in results:
        d.callback('192.0.2.101')
    d1.errback(TimeoutError())

    def checkHints(res):
        self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 12)
    d1.addBoth(checkHints)