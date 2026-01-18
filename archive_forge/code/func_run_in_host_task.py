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
def run_in_host_task(self, token: TrioToken) -> None:
    task_register = PARENT_TASK_DATA.task_register

    def in_trio_thread() -> None:
        task = task_register[0]
        assert task is not None, 'guaranteed by abandon_on_cancel semantics'
        trio.lowlevel.reschedule(task, outcome.Value(self))
    token.run_sync_soon(in_trio_thread)