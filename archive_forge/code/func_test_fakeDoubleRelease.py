import gc
import weakref
from threading import ThreadError, local
from twisted.trial.unittest import SynchronousTestCase
from .. import AlreadyQuit, LockWorker, ThreadWorker
def test_fakeDoubleRelease(self):
    """
        The L{FakeLock} test fixture will alert us if there's a potential
        double-release.
        """
    lock = FakeLock()
    self.assertRaises(ThreadError, lock.release)
    lock.acquire()
    self.assertEqual(None, lock.release())
    self.assertRaises(ThreadError, lock.release)