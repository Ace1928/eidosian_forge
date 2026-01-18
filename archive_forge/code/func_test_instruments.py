from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instruments(recwarn: object) -> None:
    r1 = TaskRecorder()
    r2 = TaskRecorder()
    r3 = TaskRecorder()
    task = None

    async def task_fn() -> None:
        nonlocal task
        task = _core.current_task()
        for _ in range(4):
            await _core.checkpoint()
        _core.remove_instrument(r2)
        with pytest.raises(KeyError):
            _core.remove_instrument(r2)
        _core.add_instrument(r3)
        _core.add_instrument(r3)
        for _ in range(1):
            await _core.checkpoint()

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(task_fn)
    _core.run(main, instruments=[r1, r2])
    expected = [('before_run', None), ('schedule', task)] + [('before', task), ('schedule', task), ('after', task)] * 5 + [('before', task), ('after', task), ('after_run', None)]
    assert r1.record == r2.record + r3.record
    assert task is not None
    assert list(r1.filter_tasks([task])) == expected