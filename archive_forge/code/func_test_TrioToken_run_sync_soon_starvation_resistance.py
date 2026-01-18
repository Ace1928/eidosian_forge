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
def test_TrioToken_run_sync_soon_starvation_resistance() -> None:
    token: _core.TrioToken | None = None
    record: list[tuple[str, int]] = []

    def naughty_cb(i: int) -> None:
        try:
            not_none(token).run_sync_soon(naughty_cb, i + 1)
        except _core.RunFinishedError:
            record.append(('run finished', i))

    async def main() -> None:
        nonlocal token
        token = _core.current_trio_token()
        token.run_sync_soon(naughty_cb, 0)
        record.append(('starting', 0))
        for _ in range(20):
            await _core.checkpoint()
    _core.run(main)
    assert len(record) == 2
    assert record[0] == ('starting', 0)
    assert record[1][0] == 'run finished'
    assert record[1][1] >= 19