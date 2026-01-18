from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
def test_gaierror(self):
    """
        Resolving a hostname that results in C{getaddrinfo} raising a
        L{gaierror} will result in the L{IResolutionReceiver} receiving a call
        to C{resolutionComplete} with no C{addressResolved} calls in between;
        no failure is logged.
        """
    receiver = ResultHolder(self)
    resolution = self.resolver.resolveHostName(receiver, 'sample.example.com')
    self.assertIs(receiver._resolution, resolution)
    self.doThreadWork()
    self.doReactorWork()
    self.assertEqual(receiver._started, True)
    self.assertEqual(receiver._ended, True)
    self.assertEqual(receiver._addresses, [])