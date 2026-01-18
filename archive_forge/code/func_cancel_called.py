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
@property
def cancel_called(self) -> bool:
    """Readonly :class:`bool`. Records whether cancellation has been
        requested for this scope, either by an explicit call to
        :meth:`cancel` or by the deadline expiring.

        This attribute being True does *not* necessarily mean that the
        code within the scope has been, or will be, affected by the
        cancellation. For example, if :meth:`cancel` was called after
        the last checkpoint in the ``with`` block, when it's too late to
        deliver a :exc:`~trio.Cancelled` exception, then this attribute
        will still be True.

        This attribute is mostly useful for debugging and introspection.
        If you want to know whether or not a chunk of code was actually
        cancelled, then :attr:`cancelled_caught` is usually more
        appropriate.
        """
    if self._cancel_status is not None or not self._has_been_entered:
        if not self._cancel_called and current_time() >= self._deadline:
            self.cancel()
    return self._cancel_called