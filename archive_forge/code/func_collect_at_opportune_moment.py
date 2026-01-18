from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def collect_at_opportune_moment(token: _core._entry_queue.TrioToken) -> None:
    runner = _core._run.GLOBAL_RUN_CONTEXT.runner
    assert runner.system_nursery is not None
    if runner.system_nursery._closed and isinstance(runner.asyncgens.alive, weakref.WeakSet):
        saved.clear()
        record.append('final collection')
        gc_collect_harder()
        record.append('done')
    else:
        try:
            token.run_sync_soon(collect_at_opportune_moment, token)
        except _core.RunFinishedError:
            nonlocal needs_retry
            needs_retry = True