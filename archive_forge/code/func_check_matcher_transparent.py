from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_matcher_transparent() -> None:
    with RaisesGroup(Matcher(ValueError)) as e:
        ...
    _: BaseExceptionGroup[ValueError] = e.value
    assert_type(e.value, BaseExceptionGroup[ValueError])