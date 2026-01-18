from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_initialStatistics(self):
    """
        L{Team.statistics} returns an object with idleWorkerCount,
        busyWorkerCount, and backloggedWorkCount integer attributes.
        """
    stats = self.team.statistics()
    self.assertEqual(stats.idleWorkerCount, 0)
    self.assertEqual(stats.busyWorkerCount, 0)
    self.assertEqual(stats.backloggedWorkCount, 0)