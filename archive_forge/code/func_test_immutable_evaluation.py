from itertools import product
from sympy.core.relational import (Equality, Unequality)
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices import SparseMatrix
from sympy.matrices.immutable import \
from sympy.abc import x, y
from sympy.testing.pytest import raises
def test_immutable_evaluation():
    X = ImmutableMatrix(eye(3))
    A = ImmutableMatrix(3, 3, range(9))
    assert isinstance(X + A, ImmutableMatrix)
    assert isinstance(X * A, ImmutableMatrix)
    assert isinstance(X * 2, ImmutableMatrix)
    assert isinstance(2 * X, ImmutableMatrix)
    assert isinstance(A ** 2, ImmutableMatrix)