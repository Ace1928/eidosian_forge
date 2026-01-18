from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_growCreateLimit(self):
    """
        L{Team.grow} increases the number of available idle workers until the
        C{createWorker} callable starts returning None.
        """
    self.noMoreWorkers = lambda: len(self.allWorkersEver) >= 3
    self.team.grow(5)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.allWorkersEver), 3)
    self.assertEqual(self.team.statistics().idleWorkerCount, 3)