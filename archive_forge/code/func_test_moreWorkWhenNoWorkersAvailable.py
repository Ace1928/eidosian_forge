from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_moreWorkWhenNoWorkersAvailable(self):
    """
        When no additional workers are available, the given work is backlogged,
        and then performed later when the work was.
        """
    self.team.grow(3)
    self.coordinate()

    def something():
        something.times += 1
    something.times = 0
    self.assertEqual(self.team.statistics().idleWorkerCount, 3)
    for i in range(3):
        self.team.do(something)
    self.coordinate()
    self.assertEqual(self.team.statistics().idleWorkerCount, 0)
    self.noMoreWorkers = lambda: True
    self.team.do(something)
    self.coordinate()
    self.assertEqual(self.team.statistics().idleWorkerCount, 0)
    self.assertEqual(self.team.statistics().backloggedWorkCount, 1)
    self.performAllOutstandingWork()
    self.assertEqual(self.team.statistics().backloggedWorkCount, 0)
    self.assertEqual(something.times, 4)