import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_doUnwindsReentrancy(self):
    """
        If L{LockWorker.do} is called recursively, it postpones the inner call
        until the outer one is complete.
        """
    lock = FakeLock()
    worker = LockWorker(lock, local())
    levels = []
    acquired = []

    def work():
        work.level += 1
        levels.append(work.level)
        acquired.append(lock.acquired)
        if len(levels) < 2:
            worker.do(work)
        work.level -= 1
    work.level = 0
    worker.do(work)
    self.assertEqual(levels, [1, 1])
    self.assertEqual(acquired, [True, True])