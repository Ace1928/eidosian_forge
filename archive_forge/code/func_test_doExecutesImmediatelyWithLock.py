import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_doExecutesImmediatelyWithLock(self):
    """
        L{LockWorker.do} immediately performs the work it's given, while the
        lock is acquired.
        """
    storage = local()
    lock = FakeLock()
    worker = LockWorker(lock, storage)

    def work():
        work.done = True
        work.acquired = lock.acquired
    work.done = False
    worker.do(work)
    self.assertEqual(work.done, True)
    self.assertEqual(work.acquired, True)
    self.assertEqual(lock.acquired, False)