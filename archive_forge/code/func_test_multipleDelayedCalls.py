import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_multipleDelayedCalls(self):
    """
        If there are several delayed calls, C{doIteration} is called with a
        timeout which is the difference between the current time and the
        time at which the earlier of the two calls is to run.
        """
    reactor = TimeoutReportReactor()
    reactor.callLater(50, lambda: None)
    reactor.callLater(10, lambda: None)
    reactor.callLater(100, lambda: None)
    timeout = self._checkIterationTimeout(reactor)
    self.assertEqual(timeout, 10)