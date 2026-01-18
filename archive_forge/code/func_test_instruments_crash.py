from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instruments_crash(caplog: pytest.LogCaptureFixture) -> None:
    record = []

    class BrokenInstrument(_abc.Instrument):

        def task_scheduled(self, task: Task) -> NoReturn:
            record.append('scheduled')
            raise ValueError('oops')

        def close(self) -> None:
            record.append('closed')

    async def main() -> Task:
        record.append('main ran')
        return _core.current_task()
    r = TaskRecorder()
    main_task = _core.run(main, instruments=[r, BrokenInstrument()])
    assert record == ['scheduled', 'main ran']
    assert ('after', main_task) in r.record
    assert ('after_run', None) in r.record
    assert caplog.records[0].exc_info is not None
    exc_type, exc_value, exc_traceback = caplog.records[0].exc_info
    assert exc_type is ValueError
    assert str(exc_value) == 'oops'
    assert 'Instrument has been disabled' in caplog.records[0].message