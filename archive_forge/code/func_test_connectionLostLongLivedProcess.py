import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_connectionLostLongLivedProcess(self):
    """
        L{ProcessMonitor.connectionLost} should immediately restart a process
        if it has been running longer than L{ProcessMonitor.threshold} seconds.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(0)
    self.assertIn('foo', self.pm.protocols)
    self.reactor.advance(self.pm.threshold)
    self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
    self.assertNotIn('foo', self.pm.protocols)
    self.reactor.advance(0)
    self.assertIn('foo', self.pm.protocols)