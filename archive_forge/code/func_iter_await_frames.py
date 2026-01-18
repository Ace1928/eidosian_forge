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
def iter_await_frames(self) -> Iterator[tuple[types.FrameType, int]]:
    """Iterates recursively over the coroutine-like objects this
        task is waiting on, yielding the frame and line number at each
        frame.

        This is similar to `traceback.walk_stack` in a synchronous
        context. Note that `traceback.walk_stack` returns frames from
        the bottom of the call stack to the top, while this function
        starts from `Task.coro <trio.lowlevel.Task.coro>` and works it
        way down.

        Example usage: extracting a stack trace::

            import traceback

            def print_stack_for_task(task):
                ss = traceback.StackSummary.extract(task.iter_await_frames())
                print("".join(ss.format()))

        """
    coro: Any = self.coro
    while coro is not None:
        if hasattr(coro, 'cr_frame'):
            yield (coro.cr_frame, coro.cr_frame.f_lineno)
            coro = coro.cr_await
        elif hasattr(coro, 'gi_frame'):
            yield (coro.gi_frame, coro.gi_frame.f_lineno)
            coro = coro.gi_yieldfrom
        elif coro.__class__.__name__ in ['async_generator_athrow', 'async_generator_asend']:
            for referent in gc.get_referents(coro):
                if hasattr(referent, 'ag_frame'):
                    yield (referent.ag_frame, referent.ag_frame.f_lineno)
                    coro = referent.ag_await
                    break
            else:
                return
        else:
            return