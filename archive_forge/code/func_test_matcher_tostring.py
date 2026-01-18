from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test_matcher_tostring() -> None:
    assert str(Matcher(ValueError)) == 'Matcher(ValueError)'
    assert str(Matcher(match='[a-z]')) == "Matcher(match='[a-z]')"
    pattern_no_flags = re.compile('noflag', 0)
    assert str(Matcher(match=pattern_no_flags)) == "Matcher(match='noflag')"
    pattern_flags = re.compile('noflag', re.IGNORECASE)
    assert str(Matcher(match=pattern_flags)) == f'Matcher(match={pattern_flags!r})'
    assert str(Matcher(ValueError, match='re', check=bool)) == f"Matcher(ValueError, match='re', check={bool!r})"