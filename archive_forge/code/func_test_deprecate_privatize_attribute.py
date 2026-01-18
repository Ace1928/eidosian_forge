from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
def test_deprecate_privatize_attribute() -> None:

    class C:

        def __init__(self) -> None:
            self._attr = 1

        def _meth(self, arg: T) -> T:
            return arg
        attr: int = _api.deprecate_privatize_attribute('0.0')
        meth: Callable = _api.deprecate_privatize_attribute('0.0')
    c = C()
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.attr == 1
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        c.attr = 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.attr == 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.meth(42) == 42