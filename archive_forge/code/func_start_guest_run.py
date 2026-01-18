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
def start_guest_run(async_fn: Callable[..., Awaitable[RetT]], *args: object, run_sync_soon_threadsafe: Callable[[Callable[[], object]], object], done_callback: Callable[[outcome.Outcome[RetT]], object], run_sync_soon_not_threadsafe: Callable[[Callable[[], object]], object] | None=None, host_uses_signal_set_wakeup_fd: bool=False, clock: Clock | None=None, instruments: Sequence[Instrument]=(), restrict_keyboard_interrupt_to_checkpoints: bool=False, strict_exception_groups: bool=True) -> None:
    """Start a "guest" run of Trio on top of some other "host" event loop.

    Each host loop can only have one guest run at a time.

    You should always let the Trio run finish before stopping the host loop;
    if not, it may leave Trio's internal data structures in an inconsistent
    state. You might be able to get away with it if you immediately exit the
    program, but it's safest not to go there in the first place.

    Generally, the best way to do this is wrap this in a function that starts
    the host loop and then immediately starts the guest run, and then shuts
    down the host when the guest run completes.

    Once :func:`start_guest_run` returns successfully, the guest run
    has been set up enough that you can invoke sync-colored Trio
    functions such as :func:`~trio.current_time`, :func:`spawn_system_task`,
    and :func:`current_trio_token`. If a `~trio.TrioInternalError` occurs
    during this early setup of the guest run, it will be raised out of
    :func:`start_guest_run`.  All other errors, including all errors
    raised by the *async_fn*, will be delivered to your
    *done_callback* at some point after :func:`start_guest_run` returns
    successfully.

    Args:

      run_sync_soon_threadsafe: An arbitrary callable, which will be passed a
         function as its sole argument::

            def my_run_sync_soon_threadsafe(fn):
                ...

         This callable should schedule ``fn()`` to be run by the host on its
         next pass through its loop. **Must support being called from
         arbitrary threads.**

      done_callback: An arbitrary callable::

            def my_done_callback(run_outcome):
                ...

         When the Trio run has finished, Trio will invoke this callback to let
         you know. The argument is an `outcome.Outcome`, reporting what would
         have been returned or raised by `trio.run`. This function can do
         anything you want, but commonly you'll want it to shut down the
         host loop, unwrap the outcome, etc.

      run_sync_soon_not_threadsafe: Like ``run_sync_soon_threadsafe``, but
         will only be called from inside the host loop's main thread.
         Optional, but if your host loop allows you to implement this more
         efficiently than ``run_sync_soon_threadsafe`` then passing it will
         make things a bit faster.

      host_uses_signal_set_wakeup_fd (bool): Pass `True` if your host loop
         uses `signal.set_wakeup_fd`, and `False` otherwise. For more details,
         see :ref:`guest-run-implementation`.

    For the meaning of other arguments, see `trio.run`.

    """
    if strict_exception_groups is not None and (not strict_exception_groups):
        warn_deprecated('trio.start_guest_run(..., strict_exception_groups=False)', version='0.24.1', issue=2929, instead='the default value of True and rewrite exception handlers to handle ExceptionGroups')
    runner = setup_runner(clock, instruments, restrict_keyboard_interrupt_to_checkpoints, strict_exception_groups)
    runner.is_guest = True
    runner.guest_tick_scheduled = True
    if run_sync_soon_not_threadsafe is None:
        run_sync_soon_not_threadsafe = run_sync_soon_threadsafe
    guest_state = GuestState(runner=runner, run_sync_soon_threadsafe=run_sync_soon_threadsafe, run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe, done_callback=done_callback, unrolled_run_gen=unrolled_run(runner, async_fn, args, host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd))
    next_send = cast(EventResult, None)
    for _tick in range(5):
        if runner.system_nursery is not None:
            break
        try:
            timeout = guest_state.unrolled_run_gen.send(next_send)
        except StopIteration:
            raise TrioInternalError('Guest runner exited before system nursery was initialized') from None
        if timeout != 0:
            guest_state.unrolled_run_gen.throw(TrioInternalError('Guest runner blocked before system nursery was initialized'))
        if sys.platform == 'win32':
            next_send = 0
        else:
            next_send = []
    else:
        guest_state.unrolled_run_gen.throw(TrioInternalError('Guest runner yielded too many times before system nursery was initialized'))
    guest_state.unrolled_run_next_send = Value(next_send)
    run_sync_soon_not_threadsafe(guest_state.guest_tick)