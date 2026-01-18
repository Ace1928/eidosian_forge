import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_addProcessCwd(self):
    """
        L{ProcessMonitor.addProcess} takes an C{cwd} parameter that is passed
        to L{IReactorProcess.spawnProcess}.
        """
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'], cwd='/mnt/lala')
    self.reactor.advance(0)
    self.assertEqual(self.reactor.spawnedProcesses[0]._path, '/mnt/lala')