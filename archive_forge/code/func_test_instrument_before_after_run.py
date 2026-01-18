from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def test_instrument_before_after_run() -> None:
    record = []

    class BeforeAfterRun(_abc.Instrument):

        def before_run(self) -> None:
            record.append('before_run')

        def after_run(self) -> None:
            record.append('after_run')

    async def main() -> None:
        pass
    _core.run(main, instruments=[BeforeAfterRun()])
    assert record == ['before_run', 'after_run']