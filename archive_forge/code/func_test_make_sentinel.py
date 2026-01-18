import re
import sys
import traceback
from typing import NoReturn
import pytest
from .._util import (
def test_make_sentinel() -> None:

    class S(Sentinel, metaclass=Sentinel):
        pass
    assert repr(S) == 'S'
    assert S == S
    assert type(S).__name__ == 'S'
    assert S in {S}
    assert type(S) is S

    class S2(Sentinel, metaclass=Sentinel):
        pass
    assert repr(S2) == 'S2'
    assert S != S2
    assert S not in {S2}
    assert type(S) is not type(S2)