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
def test_async_function_implemented_in_C() -> None:

    async def agen_fn(record: list[str]) -> AsyncIterator[None]:
        assert not _core.currently_ki_protected()
        record.append('the generator ran')
        yield
    run_record: list[str] = []
    agen = agen_fn(run_record)
    _core.run(agen.__anext__)
    assert run_record == ['the generator ran']

    async def main() -> None:
        start_soon_record: list[str] = []
        agen = agen_fn(start_soon_record)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(agen.__anext__)
        assert start_soon_record == ['the generator ran']
    _core.run(main)