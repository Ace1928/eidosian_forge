from __future__ import annotations
import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar
import trio
from ._core import (
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error
def from_thread_check_cancelled() -> None:
    """Raise `trio.Cancelled` if the associated Trio task entered a cancelled status.

     Only applicable to threads spawned by `trio.to_thread.run_sync`. Poll to allow
     ``abandon_on_cancel=False`` threads to raise :exc:`~trio.Cancelled` at a suitable
     place, or to end abandoned ``abandon_on_cancel=True`` threads sooner than they may
     otherwise.

    Raises:
        Cancelled: If the corresponding call to `trio.to_thread.run_sync` has had a
            delivery of cancellation attempted against it, regardless of the value of
            ``abandon_on_cancel`` supplied as an argument to it.
        RuntimeError: If this thread is not spawned from `trio.to_thread.run_sync`.

    .. note::

       To be precise, :func:`~trio.from_thread.check_cancelled` checks whether the task
       running :func:`trio.to_thread.run_sync` has ever been cancelled since the last
       time it was running a :func:`trio.from_thread.run` or :func:`trio.from_thread.run_sync`
       function. It may raise `trio.Cancelled` even if a cancellation occurred that was
       later hidden by a modification to `trio.CancelScope.shield` between the cancelled
       `~trio.CancelScope` and :func:`trio.to_thread.run_sync`. This differs from the
       behavior of normal Trio checkpoints, which raise `~trio.Cancelled` only if the
       cancellation is still active when the checkpoint executes. The distinction here is
       *exceedingly* unlikely to be relevant to your application, but we mention it
       for completeness.
    """
    try:
        raise_cancel = PARENT_TASK_DATA.cancel_register[0]
    except AttributeError:
        raise RuntimeError("this thread wasn't created by Trio, can't check for cancellation") from None
    if raise_cancel is not None:
        raise_cancel()