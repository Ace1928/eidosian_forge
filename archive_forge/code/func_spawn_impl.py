from __future__ import annotations
import enum
import functools
import gc
import itertools
import random
import select
import sys
import threading
import warnings
from collections import deque
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from contextvars import copy_context
from heapq import heapify, heappop, heappush
from math import inf
from time import perf_counter
from typing import (
import attrs
from outcome import Error, Outcome, Value, capture
from sniffio import thread_local as sniffio_library
from sortedcontainers import SortedDict
from .. import _core
from .._abc import Clock, Instrument
from .._deprecate import warn_deprecated
from .._util import NoPublicConstructor, coroutine_or_error, final
from ._asyncgens import AsyncGenerators
from ._concat_tb import concat_tb
from ._entry_queue import EntryQueue, TrioToken
from ._exceptions import Cancelled, RunFinishedError, TrioInternalError
from ._instrumentation import Instruments
from ._ki import LOCALS_KEY_KI_PROTECTION_ENABLED, KIManager, enable_ki_protection
from ._thread_cache import start_thread_soon
from ._traps import (
from ._generated_instrumentation import *
from ._generated_run import *
def spawn_impl(self, async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]], args: tuple[Unpack[PosArgT]], nursery: Nursery | None, name: object, *, system_task: bool=False, context: contextvars.Context | None=None) -> Task:
    if nursery is not None and nursery._closed:
        raise RuntimeError('Nursery is closed to new arrivals')
    if nursery is None:
        assert self.init_task is None
    if context is None:
        context = self.system_context.copy() if system_task else copy_context()
    coro = context.run(coroutine_or_error, async_fn, *args)
    if name is None:
        name = async_fn
    if isinstance(name, functools.partial):
        name = name.func
    if not isinstance(name, str):
        try:
            name = f'{name.__module__}.{name.__qualname__}'
        except AttributeError:
            name = repr(name)
    if getattr(coro, 'cr_frame', None) is None:

        async def python_wrapper(orig_coro: Awaitable[RetT]) -> RetT:
            return await orig_coro
        coro = python_wrapper(coro)
    coro.cr_frame.f_locals.setdefault(LOCALS_KEY_KI_PROTECTION_ENABLED, system_task)
    task = Task._create(coro=coro, parent_nursery=nursery, runner=self, name=name, context=context)
    self.tasks.add(task)
    if nursery is not None:
        nursery._children.add(task)
        task._activate_cancel_status(nursery._cancel_status)
    if 'task_spawned' in self.instruments:
        self.instruments.call('task_spawned', task)
    self.reschedule(task, None)
    return task