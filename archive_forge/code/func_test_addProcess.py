import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_addProcess(self):
    """
        L{ProcessMonitor.addProcess} only starts the named program if
        L{ProcessMonitor.startService} has been called.
        """
    self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
    self.assertEqual(self.pm.protocols, {})
    self.assertEqual(self.pm.processes, {'foo': (['arg1', 'arg2'], 1, 2, {})})
    self.pm.startService()
    self.reactor.advance(0)
    self.assertEqual(list(self.pm.protocols.keys()), ['foo'])