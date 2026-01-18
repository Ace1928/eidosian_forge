import threading
import weakref
import warnings
from inspect import iscoroutinefunction
from functools import wraps
from queue import SimpleQueue
from twisted.python import threadable
from twisted.python.runtime import platform
from twisted.python.failure import Failure
from twisted.python.log import PythonLoggingObserver, err
from twisted.internet.defer import maybeDeferred, ensureDeferred
from twisted.internet.task import LoopingCall
import wrapt
from ._util import synchronized
from ._resultstore import ResultStore
def run_in_reactor(self, function):
    """
        A decorator that ensures the wrapped function runs in the
        reactor thread.

        When the wrapped function is called, an EventualResult is returned.
        """

    def _run_in_reactor(wrapped, _, args, kwargs):
        """
            Implementation: A decorator that ensures the wrapped function runs in
            the reactor thread.

            When the wrapped function is called, an EventualResult is returned.
            """
        if iscoroutinefunction(wrapped):

            def runs_in_reactor(result, args, kwargs):
                d = ensureDeferred(wrapped(*args, **kwargs))
                result._connect_deferred(d)
        else:

            def runs_in_reactor(result, args, kwargs):
                d = maybeDeferred(wrapped, *args, **kwargs)
                result._connect_deferred(d)
        result = EventualResult(None, self._reactor)
        self._registry.register(result)
        self._reactor.callFromThread(runs_in_reactor, result, args, kwargs)
        return result
    if iscoroutinefunction(function):

        @wraps(function)
        def non_async_wrapper():
            pass
    else:
        non_async_wrapper = None
    return wrapt.decorator(_run_in_reactor, adapter=non_async_wrapper)(function)