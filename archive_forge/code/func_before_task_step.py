from __future__ import annotations
from typing import TYPE_CHECKING, Container, Iterable, NoReturn
import attrs
import pytest
from ... import _abc, _core
from .tutil import check_sequence_matches
def before_task_step(self, task: Task) -> None:
    assert task is _core.current_task()
    self.record.append(('before', task))