from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def test_delegation_to_existing_hooks() -> None:
    record = []

    def my_firstiter(agen: AsyncGenerator[object, NoReturn]) -> None:
        record.append('firstiter ' + agen.ag_frame.f_locals['arg'])

    def my_finalizer(agen: AsyncGenerator[object, NoReturn]) -> None:
        record.append('finalizer ' + agen.ag_frame.f_locals['arg'])

    async def example(arg: str) -> AsyncGenerator[int, None]:
        try:
            yield 42
        finally:
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            record.append('trio collected ' + arg)

    async def async_main() -> None:
        await step_outside_async_context(example('theirs'))
        assert await example('ours').asend(None) == 42
        gc_collect_harder()
        assert record == ['firstiter theirs', 'finalizer theirs']
        record[:] = []
        await _core.wait_all_tasks_blocked()
        assert record == ['trio collected ours']
    with restore_unraisablehook():
        old_hooks = sys.get_asyncgen_hooks()
        sys.set_asyncgen_hooks(my_firstiter, my_finalizer)
        try:
            _core.run(async_main)
        finally:
            assert sys.get_asyncgen_hooks() == (my_firstiter, my_finalizer)
            sys.set_asyncgen_hooks(*old_hooks)