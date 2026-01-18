import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def run_on_executor(*args: Any, **kwargs: Any) -> Callable:
    """Decorator to run a synchronous method asynchronously on an executor.

    Returns a future.

    The executor to be used is determined by the ``executor``
    attributes of ``self``. To use a different attribute name, pass a
    keyword argument to the decorator::

        @run_on_executor(executor='_thread_pool')
        def foo(self):
            pass

    This decorator should not be confused with the similarly-named
    `.IOLoop.run_in_executor`. In general, using ``run_in_executor``
    when *calling* a blocking method is recommended instead of using
    this decorator when *defining* a method. If compatibility with older
    versions of Tornado is required, consider defining an executor
    and using ``executor.submit()`` at the call site.

    .. versionchanged:: 4.2
       Added keyword arguments to use alternative attributes.

    .. versionchanged:: 5.0
       Always uses the current IOLoop instead of ``self.io_loop``.

    .. versionchanged:: 5.1
       Returns a `.Future` compatible with ``await`` instead of a
       `concurrent.futures.Future`.

    .. deprecated:: 5.1

       The ``callback`` argument is deprecated and will be removed in
       6.0. The decorator itself is discouraged in new code but will
       not be removed in 6.0.

    .. versionchanged:: 6.0

       The ``callback`` argument was removed.
    """

    def run_on_executor_decorator(fn: Callable) -> Callable[..., Future]:
        executor = kwargs.get('executor', 'executor')

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Future:
            async_future = Future()
            conc_future = getattr(self, executor).submit(fn, self, *args, **kwargs)
            chain_future(conc_future, async_future)
            return async_future
        return wrapper
    if args and kwargs:
        raise ValueError('cannot combine positional and keyword args')
    if len(args) == 1:
        return run_on_executor_decorator(args[0])
    elif len(args) != 0:
        raise ValueError('expected 1 argument, got %d', len(args))
    return run_on_executor_decorator