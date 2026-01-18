from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_growCreatesIdleWorkers(self):
    """
        L{Team.grow} increases the number of available idle workers.
        """
    self.team.grow(5)
    self.performAllOutstandingWork()
    self.assertEqual(len(self.workerPerformers), 5)