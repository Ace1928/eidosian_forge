from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_multipledispatch():

    @dispatch(int, int)
    def f(x, y):
        return x + y

    @dispatch(float, float)
    def f(x, y):
        return x - y
    assert f(1, 2) == 3
    assert f(1.0, 2.0) == -1.0