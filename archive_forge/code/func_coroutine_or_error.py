from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def coroutine_or_error(async_fn: Callable[[Unpack[PosArgsT]], Awaitable[RetT]], *args: Unpack[PosArgsT]) -> collections.abc.Coroutine[object, NoReturn, RetT]:

    def _return_value_looks_like_wrong_library(value: object) -> bool:
        if isinstance(value, collections.abc.Generator):
            return True
        if getattr(value, '_asyncio_future_blocking', None) is not None:
            return True
        if value.__class__.__name__ in ('Future', 'Deferred'):
            return True
        return False
    prev_loop, sniffio_loop.name = (sniffio_loop.name, 'trio')
    try:
        coro = async_fn(*args)
    except TypeError:
        if isinstance(async_fn, collections.abc.Coroutine):
            async_fn.close()
            raise TypeError(f'Trio was expecting an async function, but instead it got a coroutine object {async_fn!r}\n\nProbably you did something like:\n\n  trio.run({async_fn.__name__}(...))            # incorrect!\n  nursery.start_soon({async_fn.__name__}(...))  # incorrect!\n\nInstead, you want (notice the parentheses!):\n\n  trio.run({async_fn.__name__}, ...)            # correct!\n  nursery.start_soon({async_fn.__name__}, ...)  # correct!') from None
        if _return_value_looks_like_wrong_library(async_fn):
            raise TypeError(f"Trio was expecting an async function, but instead it got {async_fn!r} – are you trying to use a library written for asyncio/twisted/tornado or similar? That won't work without some sort of compatibility shim.") from None
        raise
    finally:
        sniffio_loop.name = prev_loop
    if not isinstance(coro, collections.abc.Coroutine):
        if _return_value_looks_like_wrong_library(coro):
            raise TypeError(f"Trio got unexpected {coro!r} – are you trying to use a library written for asyncio/twisted/tornado or similar? That won't work without some sort of compatibility shim.")
        if inspect.isasyncgen(coro):
            raise TypeError(f'start_soon expected an async function but got an async generator {coro!r}')
        raise TypeError('Trio expected an async function, but {!r} appears to be synchronous'.format(getattr(async_fn, '__qualname__', async_fn)))
    return coro