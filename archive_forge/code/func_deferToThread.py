from __future__ import annotations
import queue as Queue
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.interfaces import IReactorFromThreads
from twisted.python import failure
from twisted.python.threadpool import ThreadPool
def deferToThread(f, *args, **kwargs):
    """
    Run a function in a thread and return the result as a Deferred.

    @param f: The function to call.
    @param args: positional arguments to pass to f.
    @param kwargs: keyword arguments to pass to f.

    @return: A Deferred which fires a callback with the result of f,
    or an errback with a L{twisted.python.failure.Failure} if f throws
    an exception.
    """
    from twisted.internet import reactor
    return deferToThreadPool(reactor, reactor.getThreadPool(), f, *args, **kwargs)