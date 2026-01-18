from __future__ import annotations
import re
import sys
from types import TracebackType
from typing import Any
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
def test_Matcher_check() -> None:

    def check_oserror_and_errno_is_5(e: BaseException) -> bool:
        return isinstance(e, OSError) and e.errno == 5
    with RaisesGroup(Matcher(check=check_oserror_and_errno_is_5)):
        raise ExceptionGroup('', (OSError(5, ''),))

    def check_errno_is_5(e: OSError) -> bool:
        return e.errno == 5
    with RaisesGroup(Matcher(OSError, check=check_errno_is_5)):
        raise ExceptionGroup('', (OSError(5, ''),))
    with pytest.raises(ExceptionGroup):
        with RaisesGroup(Matcher(OSError, check=check_errno_is_5)):
            raise ExceptionGroup('', (OSError(6, ''),))