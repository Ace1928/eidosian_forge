from __future__ import annotations
import sys
import threading
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager
from inspect import isawaitable
from types import TracebackType
from typing import (
from ._core import _eventloop
from ._core._eventloop import get_async_backend, get_cancelled_exc_class, threadlocals
from ._core._synchronization import Event
from ._core._tasks import CancelScope, create_task_group
from .abc import AsyncBackend
from .abc._tasks import TaskStatus
def start_task_soon(self, func: Callable[[Unpack[PosArgsT]], Awaitable[T_Retval] | T_Retval], *args: Unpack[PosArgsT], name: object=None) -> Future[T_Retval]:
    """
        Start a task in the portal's task group.

        The task will be run inside a cancel scope which can be cancelled by cancelling
        the returned future.

        :param func: the target function
        :param args: positional arguments passed to ``func``
        :param name: name of the task (will be coerced to a string if not ``None``)
        :return: a future that resolves with the return value of the callable if the
            task completes successfully, or with the exception raised in the task
        :raises RuntimeError: if the portal is not running or if this method is called
            from within the event loop thread
        :rtype: concurrent.futures.Future[T_Retval]

        .. versionadded:: 3.0

        """
    self._check_running()
    f: Future[T_Retval] = Future()
    self._spawn_task_from_thread(func, args, {}, name, f)
    return f