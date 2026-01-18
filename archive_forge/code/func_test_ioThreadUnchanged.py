from __future__ import absolute_import
from twisted.trial.unittest import SynchronousTestCase
import threading
from twisted.python import threadable
from .._eventloop import ThreadLogObserver
def test_ioThreadUnchanged(self):
    """
        ThreadLogObserver does not change the Twisted I/O thread (which is
        supposed to match the thread the main reactor is running in.)
        """
    threadLog = ThreadLogObserver(None)
    threadLog.stop()
    threadLog._thread.join()
    self.assertIn(threadable.ioThread, (None, threading.current_thread().ident))