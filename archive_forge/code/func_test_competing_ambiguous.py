from __future__ import annotations
from typing import Any
from sympy.multipledispatch import dispatch
from sympy.multipledispatch.conflict import AmbiguityWarning
from sympy.testing.pytest import raises, warns
from functools import partial
def test_competing_ambiguous():
    test_namespace = {}
    dispatch = partial(orig_dispatch, namespace=test_namespace)

    @dispatch(A, C)
    def f(x, y):
        return 2
    with warns(AmbiguityWarning, test_stacklevel=False):

        @dispatch(C, A)
        def f(x, y):
            return 2
    assert f(A(), C()) == f(C(), A()) == 2