from __future__ import annotations
import sys
from typing import Union
from trio.testing import Matcher, RaisesGroup
from typing_extensions import assert_type
def check_nested_raisesgroups_contextmanager() -> None:
    with RaisesGroup(RaisesGroup(ValueError)) as excinfo:
        raise ExceptionGroup('foo', (ValueError(),))
    _: BaseExceptionGroup[BaseExceptionGroup[ValueError]] = excinfo.value
    print(excinfo.value.exceptions[0].exceptions[0])
    print(type(excinfo.value))
    assert_type(excinfo.value, BaseExceptionGroup[RaisesGroup[ValueError]])
    print(type(excinfo.value.exceptions[0]))
    assert_type(excinfo.value.exceptions[0], Union[RaisesGroup[ValueError], BaseExceptionGroup[RaisesGroup[ValueError]]])