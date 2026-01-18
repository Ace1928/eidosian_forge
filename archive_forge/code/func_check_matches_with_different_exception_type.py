from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_matches_with_different_exception_type() -> None:
    e: BaseExceptionGroup[KeyboardInterrupt] = BaseExceptionGroup('', (KeyboardInterrupt(),))
    if RaisesGroup(ValueError).matches(e):
        assert_type(e, BaseExceptionGroup[ValueError])