from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_caching_correct_behavior():

    @dispatch(A)
    def f(x):
        return 1
    assert f(C()) == 1

    @dispatch(C)
    def f(x):
        return 2
    assert f(C()) == 2