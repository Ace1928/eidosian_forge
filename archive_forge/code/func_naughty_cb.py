from __future__ import annotations
import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, cast
import outcome
import pytest
import sniffio
from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD
from .tutil import (
def naughty_cb(i: int) -> None:
    try:
        not_none(token).run_sync_soon(naughty_cb, i + 1)
    except _core.RunFinishedError:
        record.append(('run finished', i))