from collections import defaultdict
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.numbers import Integer
from sympy.core.kind import NumberKind
from sympy.matrices.common import MatrixKind
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.sets.sets import FiniteSet
from sympy.core.containers import tuple_wrapper, TupleKind
from sympy.core.expr import unchanged
from sympy.core.function import Function, Lambda
from sympy.core.relational import Eq
from sympy.testing.pytest import raises
from sympy.utilities.iterables import is_sequence, iterable
from sympy.abc import x, y
def test_Tuple_Eq():
    assert Eq(Tuple(), Tuple()) is S.true
    assert Eq(Tuple(1), 1) is S.false
    assert Eq(Tuple(1, 2), Tuple(1)) is S.false
    assert Eq(Tuple(1), Tuple(1)) is S.true
    assert Eq(Tuple(1, 2), Tuple(1, 3)) is S.false
    assert Eq(Tuple(1, 2), Tuple(1, 2)) is S.true
    assert unchanged(Eq, Tuple(1, x), Tuple(1, 2))
    assert Eq(Tuple(1, x), Tuple(1, 2)).subs(x, 2) is S.true
    assert unchanged(Eq, Tuple(1, 2), x)
    f = Function('f')
    assert unchanged(Eq, Tuple(1), f(x))
    assert Eq(Tuple(1), f(x)).subs(x, 1).subs(f, Lambda(y, (y,))) is S.true