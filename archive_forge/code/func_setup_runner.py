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
def setup_runner(clock: Clock | None, instruments: Sequence[Instrument], restrict_keyboard_interrupt_to_checkpoints: bool, strict_exception_groups: bool) -> Runner:
    """Create a Runner object and install it as the GLOBAL_RUN_CONTEXT."""
    if hasattr(GLOBAL_RUN_CONTEXT, 'runner'):
        raise RuntimeError('Attempted to call run() from inside a run()')
    if clock is None:
        clock = SystemClock()
    instrument_group = Instruments(instruments)
    io_manager = TheIOManager()
    system_context = copy_context()
    ki_manager = KIManager()
    runner = Runner(clock=clock, instruments=instrument_group, io_manager=io_manager, system_context=system_context, ki_manager=ki_manager, strict_exception_groups=strict_exception_groups)
    runner.asyncgens.install_hooks(runner)
    ki_manager.install(runner.deliver_ki, restrict_keyboard_interrupt_to_checkpoints)
    GLOBAL_RUN_CONTEXT.runner = runner
    return runner