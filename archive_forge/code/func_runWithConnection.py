from twisted.internet import threads
from twisted.python import log, reflect
def runWithConnection(self, func, *args, **kw):
    """
        Execute a function with a database connection and return the result.

        @param func: A callable object of one argument which will be executed
            in a thread with a connection from the pool. It will be passed as
            its first argument a L{Connection} instance (whose interface is
            mostly identical to that of a connection object for your DB-API
            module of choice), and its results will be returned as a
            L{Deferred}. If the method raises an exception the transaction will
            be rolled back. Otherwise, the transaction will be committed.
            B{Note} that this function is B{not} run in the main thread: it
            must be threadsafe.

        @param args: positional arguments to be passed to func

        @param kw: keyword arguments to be passed to func

        @return: a L{Deferred} which will fire the return value of
            C{func(Transaction(...), *args, **kw)}, or a
            L{twisted.python.failure.Failure}.
        """
    return threads.deferToThreadPool(self._reactor, self.threadpool, self._runWithConnection, func, *args, **kw)