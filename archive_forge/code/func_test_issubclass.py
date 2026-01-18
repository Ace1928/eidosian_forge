from __future__ import annotations
from typing import (
import pytest
import numpy as np
import numpy.typing as npt
import numpy._typing as _npt
def test_issubclass(self, cls: type[Any], obj: object) -> None:
    if cls is _npt._SupportsDType:
        pytest.xfail("Protocols with non-method members don't support issubclass()")
    assert issubclass(type(obj), cls)
    assert not issubclass(type(None), cls)