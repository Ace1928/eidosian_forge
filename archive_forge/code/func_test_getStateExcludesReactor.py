import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_getStateExcludesReactor(self):
    """
        The private L{ProcessMonitor._reactor} instance variable should not be
        included in the pickle state.
        """
    self.assertNotIn('_reactor', self.pm.__getstate__())