from . import events
from . import futures
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import typing
def get_coro(self) -> typing.Union[collections.abc.Generator, collections.abc.Coroutine]:
    return self._coro