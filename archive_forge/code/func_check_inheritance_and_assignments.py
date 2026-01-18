from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_inheritance_and_assignments() -> None:
    _: BaseExceptionGroup[ValueError] = RaisesGroup(ValueError)
    _ = RaisesGroup(RaisesGroup(ValueError))
    a: BaseExceptionGroup[BaseExceptionGroup[ValueError]]
    a = RaisesGroup(RaisesGroup(ValueError))
    a = BaseExceptionGroup('', (BaseExceptionGroup('', (ValueError(),)),))
    assert a