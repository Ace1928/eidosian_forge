import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
def run_on_executor_decorator(fn: Callable) -> Callable[..., Future]:
    executor = kwargs.get('executor', 'executor')

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Future:
        async_future = Future()
        conc_future = getattr(self, executor).submit(fn, self, *args, **kwargs)
        chain_future(conc_future, async_future)
        return async_future
    return wrapper