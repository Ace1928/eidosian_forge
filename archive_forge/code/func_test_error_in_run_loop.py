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
@restore_unraisablehook()
def test_error_in_run_loop() -> None:

    async def main() -> None:
        task = _core.current_task()
        task._schedule_points = 'hello!'
        await _core.checkpoint()
    with ignore_coroutine_never_awaited_warnings():
        with pytest.raises(_core.TrioInternalError):
            _core.run(main)