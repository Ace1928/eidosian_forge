import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callInThreadWithCallbackExceptionInCallback(self):
    """
        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a
        two-tuple of C{(False, failure)} where C{failure} represents the
        exception raised by the callable supplied.
        """

    class NewError(Exception):
        pass

    def raiseError():
        raise NewError()
    waiter = threading.Lock()
    waiter.acquire()
    results = []

    def onResult(success, result):
        waiter.release()
        results.append(success)
        results.append(result)
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThreadWithCallback(onResult, raiseError)
    tp.start()
    try:
        self._waitForLock(waiter)
    finally:
        tp.stop()
    self.assertFalse(results[0])
    self.assertIsInstance(results[1], failure.Failure)
    self.assertTrue(issubclass(results[1].type, NewError))