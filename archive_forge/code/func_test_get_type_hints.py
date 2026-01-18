from __future__ import annotations
from typing import (
import pytest
import numpy as np
import numpy.typing as npt
import numpy._typing as _npt
@pytest.mark.parametrize('name,tup', TYPES.items(), ids=TYPES.keys())
def test_get_type_hints(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints`."""
    typ = tup.typ

    def func(a):
        pass
    func.__annotations__ = {'a': typ, 'return': None}
    out = get_type_hints(func)
    ref = {'a': typ, 'return': type(None)}
    assert out == ref