import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_adjustingWhenPoolStopped(self):
    """
        L{ThreadPool.adjustPoolsize} only modifies the pool size and does not
        start new workers while the pool is not running.
        """
    pool = threadpool.ThreadPool(0, 5)
    pool.start()
    pool.stop()
    pool.adjustPoolsize(2)
    self.assertEqual(len(pool.threads), 0)