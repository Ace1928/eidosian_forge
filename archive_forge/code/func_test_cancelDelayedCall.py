import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_cancelDelayedCall(self):
    """
        If the only delayed call is canceled, L{None} is the timeout passed
        to C{doIteration}.
        """
    reactor = TimeoutReportReactor()
    call = reactor.callLater(50, lambda: None)
    call.cancel()
    timeout = self._checkIterationTimeout(reactor)
    self.assertIsNone(timeout)