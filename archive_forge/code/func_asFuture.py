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
def asFuture(self, loop: AbstractEventLoop) -> 'Future[_SelfResultT]':
    """
        Adapt this L{Deferred} into a L{Future} which is bound to C{loop}.

        @note: converting a L{Deferred} to an L{Future} consumes both
            its result and its errors, so this method implicitly converts
            C{self} into a L{Deferred} firing with L{None}, regardless of what
            its result previously would have been.

        @since: Twisted 17.5.0

        @param loop: The L{asyncio} event loop to bind the L{Future} to.

        @return: A L{Future} which will fire when the L{Deferred} fires.
        """
    future = loop.create_future()

    def checkCancel(futureAgain: 'Future[_SelfResultT]') -> None:
        if futureAgain.cancelled():
            self.cancel()

    def maybeFail(failure: Failure) -> None:
        if not future.cancelled():
            future.set_exception(failure.value)

    def maybeSucceed(result: object) -> None:
        if not future.cancelled():
            future.set_result(result)
    self.addCallbacks(maybeSucceed, maybeFail)
    future.add_done_callback(checkCancel)
    return future