from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test__ExceptionInfo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trio.testing._raises_group, 'ExceptionInfo', trio.testing._raises_group._ExceptionInfo)
    with trio.testing.RaisesGroup(ValueError) as excinfo:
        raise ExceptionGroup('', (ValueError('hello'),))
    assert excinfo.type is ExceptionGroup
    assert excinfo.value.exceptions[0].args == ('hello',)
    assert isinstance(excinfo.tb, TracebackType)