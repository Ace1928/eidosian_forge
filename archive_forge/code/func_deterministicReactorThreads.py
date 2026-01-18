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
def deterministicReactorThreads():
    """
    Create a deterministic L{IReactorThreads}

    @return: a 2-tuple consisting of an L{IReactorThreads}-like object and a
        0-argument callable that will perform one unit of work invoked via that
        object's C{callFromThread} method.
    """
    worker, doer = createMemoryWorker()

    class CFT:

        def callFromThread(self, f, *a, **k):
            worker.do(lambda: f(*a, **k))
    return (CFT(), doer)