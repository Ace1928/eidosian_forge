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
def fromFuture(cls, future: 'Future[_SelfResultT]') -> 'Deferred[_SelfResultT]':
    """
        Adapt a L{Future} to a L{Deferred}.

        @note: This creates a L{Deferred} from a L{Future}, I{not} from
            a C{coroutine}; in other words, you will need to call
            L{asyncio.ensure_future}, L{asyncio.loop.create_task} or create an
            L{asyncio.Task} yourself to get from a C{coroutine} to a
            L{Future} if what you have is an awaitable coroutine and
            not a L{Future}.  (The length of this list of techniques is
            exactly why we have left it to the caller!)

        @since: Twisted 17.5.0

        @param future: The L{Future} to adapt.

        @return: A L{Deferred} which will fire when the L{Future} fires.
        """

    def adapt(result: Future[_SelfResultT]) -> None:
        try:
            extracted: _SelfResultT | Failure = result.result()
        except BaseException:
            extracted = Failure()
        actual.callback(extracted)
    futureCancel = object()

    def cancel(reself: Deferred[object]) -> None:
        future.cancel()
        reself.callback(futureCancel)
    self = cls(cancel)
    actual = self

    def uncancel(result: _SelfResultT) -> Union[_SelfResultT, Deferred[_SelfResultT]]:
        if result is futureCancel:
            nonlocal actual
            actual = Deferred()
            return actual
        return result
    self.addCallback(uncancel)
    future.add_done_callback(adapt)
    return self