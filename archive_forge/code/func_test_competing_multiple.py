from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_competing_multiple():

    @dispatch(A, B)
    def h(x, y):
        return 1

    @dispatch(C, B)
    def h(x, y):
        return 2
    assert h(D(), B()) == 2