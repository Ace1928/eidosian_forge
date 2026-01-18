import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_isNotInIOThread(self):
    """
        The reactor registers itself as the I/O thread when it runs so that
        L{twisted.python.threadable.isInIOThread} returns C{False} if it is
        called in a different thread than the reactor is running in.
        """
    results = []
    reactor = self.buildReactor()

    def check():
        results.append(isInIOThread())
        reactor.callFromThread(reactor.stop)
    reactor.callInThread(check)
    self.runReactor(reactor)
    self.assertEqual([False], results)