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
def deterministicPool():
    """
    Create a deterministic threadpool.

    @return: 2-tuple of L{ThreadPool}, 0-argument C{work} callable; when
        C{work} is called, do the work.
    """
    worker, doer = createMemoryWorker()
    return (DeterministicThreadPool(Team(LockWorker(Lock(), local()), lambda: worker, lambda: None)), doer)