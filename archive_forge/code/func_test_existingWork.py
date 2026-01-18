import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_existingWork(self):
    """
        Work added to the threadpool before its start should be executed once
        the threadpool is started: this is ensured by trying to release a lock
        previously acquired.
        """
    waiter = threading.Lock()
    waiter.acquire()
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThread(waiter.release)
    tp.start()
    try:
        self._waitForLock(waiter)
    finally:
        tp.stop()