import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_fakeDeadlock(self):
    """
        The L{FakeLock} test fixture will alert us if there's a potential
        deadlock.
        """
    lock = FakeLock()
    lock.acquire()
    self.assertRaises(WouldDeadlock, lock.acquire)