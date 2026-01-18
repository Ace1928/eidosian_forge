from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test_RaisesGroup_matches() -> None:
    rg = RaisesGroup(ValueError)
    assert not rg.matches(None)
    assert not rg.matches(ValueError())
    assert rg.matches(ExceptionGroup('', (ValueError(),)))