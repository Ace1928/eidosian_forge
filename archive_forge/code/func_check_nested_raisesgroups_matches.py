from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_nested_raisesgroups_matches() -> None:
    """Check nested RaisesGroups with .matches"""
    exc: ExceptionGroup[ExceptionGroup[ValueError]] = ExceptionGroup('', (ExceptionGroup('', (ValueError(),)),))
    if RaisesGroup(RaisesGroup(ValueError)).matches(exc):
        assert_type(exc, BaseExceptionGroup[RaisesGroup[ValueError]])