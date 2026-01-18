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
def test_Tuple_equality():
    assert not isinstance(Tuple(1, 2), tuple)
    assert (Tuple(1, 2) == (1, 2)) is True
    assert (Tuple(1, 2) != (1, 2)) is False
    assert (Tuple(1, 2) == (1, 3)) is False
    assert (Tuple(1, 2) != (1, 3)) is True
    assert (Tuple(1, 2) == Tuple(1, 2)) is True
    assert (Tuple(1, 2) != Tuple(1, 2)) is False
    assert (Tuple(1, 2) == Tuple(1, 3)) is False
    assert (Tuple(1, 2) != Tuple(1, 3)) is True