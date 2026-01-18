import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_delayedCallFromThread(self):
    """
        A function scheduled with L{IReactorThreads.callFromThread} invoked
        from a delayed call is run immediately in the next reactor iteration.

        When invoked from the reactor thread, previous implementations of
        L{IReactorThreads.callFromThread} would skip the pipe/socket based wake
        up step, assuming the reactor would wake up on its own.  However, this
        resulted in the reactor not noticing an insert into the thread queue at
        the right time (in this case, after the thread queue has been processed
        for that reactor iteration).
        """
    reactor = self.buildReactor()

    def threadCall():
        reactor.stop()
    reactor.callLater(0, reactor.callFromThread, threadCall)
    before = reactor.seconds()
    self.runReactor(reactor, 60)
    after = reactor.seconds()
    self.assertTrue(after - before < 30)