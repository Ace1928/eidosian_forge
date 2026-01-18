from twisted.python.components import proxyForInterface
from twisted.python.context import call, get
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, IWorker, Team, createMemoryWorker
def test_doDoesWorkInWorker(self):
    """
        L{Team.do} does the work in a worker created by the createWorker
        callable.
        """

    def something():
        something.who = get('worker')
    self.team.do(something)
    self.coordinate()
    self.assertEqual(self.team.statistics().busyWorkerCount, 1)
    self.performAllOutstandingWork()
    self.assertEqual(something.who, 1)
    self.assertEqual(self.team.statistics().busyWorkerCount, 0)