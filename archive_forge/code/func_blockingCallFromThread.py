from __future__ import annotations
import queue as Queue
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.interfaces import IReactorFromThreads
from twisted.python import failure
from twisted.python.threadpool import ThreadPool
def blockingCallFromThread(reactor, f, *a, **kw):
    """
    Run a function in the reactor from a thread, and wait for the result
    synchronously.  If the function returns a L{Deferred}, wait for its
    result and return that.

    @param reactor: The L{IReactorThreads} provider which will be used to
        schedule the function call.
    @param f: the callable to run in the reactor thread
    @type f: any callable.
    @param a: the arguments to pass to C{f}.
    @param kw: the keyword arguments to pass to C{f}.

    @return: the result of the L{Deferred} returned by C{f}, or the result
        of C{f} if it returns anything other than a L{Deferred}.

    @raise Exception: If C{f} raises a synchronous exception,
        C{blockingCallFromThread} will raise that exception.  If C{f}
        returns a L{Deferred} which fires with a L{Failure},
        C{blockingCallFromThread} will raise that failure's exception (see
        L{Failure.raiseException}).
    """
    queue = Queue.Queue()

    def _callFromThread():
        result = defer.maybeDeferred(f, *a, **kw)
        result.addBoth(queue.put)
    reactor.callFromThread(_callFromThread)
    result = queue.get()
    if isinstance(result, failure.Failure):
        result.raiseException()
    return result