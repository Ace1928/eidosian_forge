from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def filter_tasks(self, tasks: Container[Task]) -> Iterable[tuple[str, Task | None]]:
    for item in self.record:
        if item[0] in ('schedule', 'before', 'after') and item[1] in tasks:
            yield item
        if item[0] in ('before_run', 'after_run'):
            yield item