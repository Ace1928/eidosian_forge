from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instruments_interleave() -> None:
    tasks = {}

    async def two_step1() -> None:
        tasks['t1'] = _core.current_task()
        await _core.checkpoint()

    async def two_step2() -> None:
        tasks['t2'] = _core.current_task()
        await _core.checkpoint()

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(two_step1)
            nursery.start_soon(two_step2)
    r = TaskRecorder()
    _core.run(main, instruments=[r])
    expected = [('before_run', None), ('schedule', tasks['t1']), ('schedule', tasks['t2']), {('before', tasks['t1']), ('schedule', tasks['t1']), ('after', tasks['t1']), ('before', tasks['t2']), ('schedule', tasks['t2']), ('after', tasks['t2'])}, {('before', tasks['t1']), ('after', tasks['t1']), ('before', tasks['t2']), ('after', tasks['t2'])}, ('after_run', None)]
    print(list(r.filter_tasks(tasks.values())))
    check_sequence_matches(list(r.filter_tasks(tasks.values())), expected)