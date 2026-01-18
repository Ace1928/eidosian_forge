import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_callFromThread(self):
    """
        A function scheduled with L{IReactorThreads.callFromThread} invoked
        from another thread is run in the reactor thread.
        """
    reactor = self.buildReactor()
    result = []

    def threadCall():
        result.append(threading.current_thread())
        reactor.stop()
    reactor.callLater(0, reactor.callInThread, reactor.callFromThread, threadCall)
    self.runReactor(reactor, 5)
    self.assertEqual(result, [threading.current_thread()])