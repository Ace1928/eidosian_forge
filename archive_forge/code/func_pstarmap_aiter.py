import abc
import collections
import contextlib
import functools
import inspect
from concurrent.futures import CancelledError
from typing import (
import duet.impl as impl
from duet.aitertools import aenumerate, aiter, AnyIterable, AsyncCollector
from duet.futuretools import AwaitableFuture
def pstarmap_aiter(self, func: Callable[..., Awaitable[U]], iterable: AnyIterable[Any]) -> AsyncIterator[U]:
    return pstarmap_aiter(self.scope, func, self.limiter.throttle(iterable))