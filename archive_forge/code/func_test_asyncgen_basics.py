from __future__ import annotations
import contextlib
import sys
import weakref
from math import inf
from typing import TYPE_CHECKING, NoReturn
import pytest
from ... import _core
from .tutil import gc_collect_harder, restore_unraisablehook
def test_asyncgen_basics() -> None:
    collected = []

    async def example(cause: str) -> AsyncGenerator[int, None]:
        try:
            with contextlib.suppress(GeneratorExit):
                yield 42
            await _core.checkpoint()
        except _core.Cancelled:
            assert 'exhausted' not in cause
            task_name = _core.current_task().name
            assert cause in task_name or task_name == '<init>'
            assert _core.current_effective_deadline() == -inf
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            collected.append(cause)
        else:
            assert 'async_main' in _core.current_task().name
            assert 'exhausted' in cause
            assert _core.current_effective_deadline() == inf
            await _core.checkpoint()
            collected.append(cause)
    saved = []

    async def async_main() -> None:
        with pytest.warns(ResourceWarning, match='Async generator.*collected before.*exhausted'):
            assert await example('abandoned').asend(None) == 42
            gc_collect_harder()
        await _core.wait_all_tasks_blocked()
        assert collected.pop() == 'abandoned'
        aiter_ = example('exhausted 1')
        try:
            assert await aiter_.asend(None) == 42
        finally:
            await aiter_.aclose()
        assert collected.pop() == 'exhausted 1'
        async for val in example('exhausted 2'):
            assert val == 42
        assert collected.pop() == 'exhausted 2'
        gc_collect_harder()
        aiter_ = example('exhausted 3')
        try:
            saved.append(aiter_)
            assert await aiter_.asend(None) == 42
        finally:
            await aiter_.aclose()
        assert collected.pop() == 'exhausted 3'
        saved.append(example('exhausted 4'))
        async for val in saved[-1]:
            assert val == 42
        assert collected.pop() == 'exhausted 4'
        saved.append(example('outlived run'))
        assert await saved[-1].asend(None) == 42
        assert collected == []
    _core.run(async_main)
    assert collected.pop() == 'outlived run'
    for agen in saved:
        assert agen.ag_frame is None