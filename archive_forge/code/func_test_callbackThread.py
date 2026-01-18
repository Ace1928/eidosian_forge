import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callbackThread(self):
    """
        L{ThreadPool.callInThreadWithCallback} calls the function it is
        given and the C{onResult} callback in the same thread.
        """
    threadIds = []
    event = threading.Event()

    def onResult(success, result):
        threadIds.append(threading.current_thread().ident)
        event.set()

    def func():
        threadIds.append(threading.current_thread().ident)
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThreadWithCallback(onResult, func)
    tp.start()
    self.addCleanup(tp.stop)
    event.wait(self.getTimeout())
    self.assertEqual(len(threadIds), 2)
    self.assertEqual(threadIds[0], threadIds[1])