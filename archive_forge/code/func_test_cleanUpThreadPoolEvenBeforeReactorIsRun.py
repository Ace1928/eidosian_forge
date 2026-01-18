import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_cleanUpThreadPoolEvenBeforeReactorIsRun(self):
    """
        When the reactor has its shutdown event fired before it is run, the
        thread pool is completely destroyed.

        For what it's worth, the reason we support this behavior at all is
        because Trial does this.

        This is the case of the thread pool being created without the reactor
        being started at al.
        """
    reactor = self.buildReactor()
    threadPoolRef = ref(reactor.getThreadPool())
    reactor.fireSystemEvent('shutdown')
    gc.collect()
    self.assertIsNone(threadPoolRef())