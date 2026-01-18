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
@contextmanager
def start_blocking_portal(backend: str='asyncio', backend_options: dict[str, Any] | None=None) -> Generator[BlockingPortal, Any, None]:
    """
    Start a new event loop in a new thread and run a blocking portal in its main task.

    The parameters are the same as for :func:`~anyio.run`.

    :param backend: name of the backend
    :param backend_options: backend options
    :return: a context manager that yields a blocking portal

    .. versionchanged:: 3.0
        Usage as a context manager is now required.

    """

    async def run_portal() -> None:
        async with BlockingPortal() as portal_:
            if future.set_running_or_notify_cancel():
                future.set_result(portal_)
                await portal_.sleep_until_stopped()
    future: Future[BlockingPortal] = Future()
    with ThreadPoolExecutor(1) as executor:
        run_future = executor.submit(_eventloop.run, run_portal, backend=backend, backend_options=backend_options)
        try:
            wait(cast(Iterable[Future], [run_future, future]), return_when=FIRST_COMPLETED)
        except BaseException:
            future.cancel()
            run_future.cancel()
            raise
        if future.done():
            portal = future.result()
            cancel_remaining_tasks = False
            try:
                yield portal
            except BaseException:
                cancel_remaining_tasks = True
                raise
            finally:
                try:
                    portal.call(portal.stop, cancel_remaining_tasks)
                except RuntimeError:
                    pass
        run_future.result()