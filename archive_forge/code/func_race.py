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
def race(ds: Sequence[Deferred[_T]]) -> Deferred[tuple[int, _T]]:
    """
    Select the first available result from the sequence of Deferreds and
    cancel the rest.

    @return: A cancellable L{Deferred} that fires with the index and output of
        the element of C{ds} to have a success result first, or that fires
        with L{FailureGroup} holding a list of their failures if they all
        fail.
    """
    winner: Optional[Deferred[_T]] = None

    def cancel(result: Deferred[_T]) -> None:
        for d in to_cancel:
            d.cancel()
    final_result: Deferred[tuple[int, _T]] = Deferred(canceller=cancel)

    def succeeded(this_output: _T, this_index: int) -> None:
        nonlocal winner
        if winner is None:
            winner = to_cancel[this_index]
            for d in to_cancel:
                if d is not winner:
                    d.cancel()
            final_result.callback((this_index, this_output))
    failure_state = []

    def failed(failure: Failure, this_index: int) -> None:
        failure_state.append((this_index, failure))
        if len(failure_state) == len(to_cancel):
            failure_state.sort()
            failures = [f for ignored, f in failure_state]
            final_result.errback(FailureGroup(failures))
    to_cancel = list(ds)
    for index, d in enumerate(ds):
        d.addCallbacks(succeeded, failed, callbackArgs=(index,), errbackArgs=(index,))
    return final_result