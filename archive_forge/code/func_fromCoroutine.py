from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
@classmethod
def fromCoroutine(cls, coro: Union[Coroutine[Deferred[Any], Any, _T], Generator[Deferred[Any], Any, _T]]) -> 'Deferred[_T]':
    """
        Schedule the execution of a coroutine that awaits on L{Deferred}s,
        wrapping it in a L{Deferred} that will fire on success/failure of the
        coroutine.

        Coroutine functions return a coroutine object, similar to how
        generators work. This function turns that coroutine into a Deferred,
        meaning that it can be used in regular Twisted code. For example::

            import treq
            from twisted.internet.defer import Deferred
            from twisted.internet.task import react

            async def crawl(pages):
                results = {}
                for page in pages:
                    results[page] = await treq.content(await treq.get(page))
                return results

            def main(reactor):
                pages = [
                    "http://localhost:8080"
                ]
                d = Deferred.fromCoroutine(crawl(pages))
                d.addCallback(print)
                return d

            react(main)

        @since: Twisted 21.2.0

        @param coro: The coroutine object to schedule.

        @raise ValueError: If C{coro} is not a coroutine or generator.
        """
    if iscoroutine(coro) or inspect.isgenerator(coro):
        return _cancellableInlineCallbacks(coro)
    raise NotACoroutineError(f'{coro!r} is not a coroutine')