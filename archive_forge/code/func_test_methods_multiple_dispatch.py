from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_methods_multiple_dispatch():

    class Foo:

        @dispatch(A, A)
        def f(x, y):
            return 1

        @dispatch(A, C)
        def f(x, y):
            return 2
    foo = Foo()
    assert foo.f(A(), A()) == 1
    assert foo.f(A(), C()) == 2
    assert foo.f(C(), C()) == 2