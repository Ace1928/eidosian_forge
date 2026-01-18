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
def guest_tick(self) -> None:
    prev_library, sniffio_library.name = (sniffio_library.name, 'trio')
    try:
        timeout = self.unrolled_run_next_send.send(self.unrolled_run_gen)
    except StopIteration:
        assert self.runner.main_task_outcome is not None
        self.done_callback(self.runner.main_task_outcome)
        return
    except TrioInternalError as exc:
        self.done_callback(Error(exc))
        return
    finally:
        sniffio_library.name = prev_library
    events_outcome: Value[EventResult] | Error = capture(self.runner.io_manager.get_events, 0)
    if timeout <= 0 or isinstance(events_outcome, Error) or events_outcome.value:
        self.unrolled_run_next_send = events_outcome
        self.runner.guest_tick_scheduled = True
        self.run_sync_soon_not_threadsafe(self.guest_tick)
    else:
        self.runner.guest_tick_scheduled = False

        def get_events() -> EventResult:
            return self.runner.io_manager.get_events(timeout)

        def deliver(events_outcome: Outcome[EventResult]) -> None:

            def in_main_thread() -> None:
                self.unrolled_run_next_send = events_outcome
                self.runner.guest_tick_scheduled = True
                self.guest_tick()
            self.run_sync_soon_threadsafe(in_main_thread)
        start_thread_soon(get_events, deliver)