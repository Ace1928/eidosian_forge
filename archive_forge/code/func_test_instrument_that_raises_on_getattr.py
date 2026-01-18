from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instrument_that_raises_on_getattr() -> None:

    class EvilInstrument(_abc.Instrument):

        def task_exited(self, task: Task) -> NoReturn:
            raise AssertionError('this should never happen')

        @property
        def after_run(self) -> NoReturn:
            raise ValueError('oops')

    async def main() -> None:
        with pytest.raises(ValueError, match='^oops$'):
            _core.add_instrument(EvilInstrument())
        runner = _core.current_task()._runner
        assert 'after_run' not in runner.instruments
        assert 'task_exited' not in runner.instruments
    _core.run(main)