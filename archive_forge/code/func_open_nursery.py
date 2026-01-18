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
def open_nursery(strict_exception_groups: bool | None=None) -> AbstractAsyncContextManager[Nursery]:
    """Returns an async context manager which must be used to create a
    new `Nursery`.

    It does not block on entry; on exit it blocks until all child tasks
    have exited.

    Args:
      strict_exception_groups (bool): Unless set to False, even a single raised exception
          will be wrapped in an exception group. If not specified, uses the value passed
          to :func:`run`, which defaults to true. Setting it to False will be deprecated
          and ultimately removed in a future version of Trio.

    """
    if strict_exception_groups is not None and (not strict_exception_groups):
        warn_deprecated('open_nursery(strict_exception_groups=False)', version='0.24.1', issue=2929, instead='the default value of True and rewrite exception handlers to handle ExceptionGroups')
    if strict_exception_groups is None:
        strict_exception_groups = GLOBAL_RUN_CONTEXT.runner.strict_exception_groups
    return NurseryManager(strict_exception_groups=strict_exception_groups)