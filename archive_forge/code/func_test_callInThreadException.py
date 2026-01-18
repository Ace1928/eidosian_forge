import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callInThreadException(self):
    """
        L{ThreadPool.callInThread} logs exceptions raised by the callable it
        is passed.
        """

    class NewError(Exception):
        pass

    def raiseError():
        raise NewError()
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThread(raiseError)
    tp.start()
    tp.stop()
    errors = self.flushLoggedErrors(NewError)
    self.assertEqual(len(errors), 1)