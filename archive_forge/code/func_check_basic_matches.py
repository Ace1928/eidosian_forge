from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_basic_matches() -> None:
    exc: ExceptionGroup[ValueError] | ValueError = ExceptionGroup('', (ValueError(),))
    if RaisesGroup(ValueError).matches(exc):
        assert_type(exc, BaseExceptionGroup[ValueError])