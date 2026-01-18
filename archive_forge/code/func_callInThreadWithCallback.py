from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
def callInThreadWithCallback(self, onResult: Optional[Callable[[bool, _R], object]], func: Callable[_P, _R], *args: _P.args, **kw: _P.kwargs) -> None:
    """
        Call a callable object in a separate thread and call C{onResult} with
        the return value, or a L{twisted.python.failure.Failure} if the
        callable raises an exception.

        The callable is allowed to block, but the C{onResult} function must not
        block and should perform as little work as possible.

        A typical action for C{onResult} for a threadpool used with a Twisted
        reactor would be to schedule a L{twisted.internet.defer.Deferred} to
        fire in the main reactor thread using C{.callFromThread}.  Note that
        C{onResult} is called inside the separate thread, not inside the
        reactor thread.

        @param onResult: a callable with the signature C{(success, result)}.
            If the callable returns normally, C{onResult} is called with
            C{(True, result)} where C{result} is the return value of the
            callable.  If the callable throws an exception, C{onResult} is
            called with C{(False, failure)}.

            Optionally, C{onResult} may be L{None}, in which case it is not
            called at all.

        @param func: callable object to be called in separate thread

        @param args: positional arguments to be passed to C{func}

        @param kw: keyword arguments to be passed to C{func}
        """
    if self.joined:
        return
    ctx = context.theContextTracker.currentContext().contexts[-1]

    def inContext() -> None:
        try:
            result = inContext.theWork()
            ok = True
        except BaseException:
            result = Failure()
            ok = False
        inContext.theWork = None
        if inContext.onResult is not None:
            inContext.onResult(ok, result)
            inContext.onResult = None
        elif not ok:
            log.err(result)
    inContext.theWork = lambda: context.call(ctx, func, *args, **kw)
    inContext.onResult = onResult
    self._team.do(inContext)