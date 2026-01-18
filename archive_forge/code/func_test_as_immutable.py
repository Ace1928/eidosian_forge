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
def test_as_immutable():
    data = [[1, 2], [3, 4]]
    X = Matrix(data)
    assert sympify(X) == X.as_immutable() == ImmutableMatrix(data)
    data = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
    X = SparseMatrix(2, 2, data)
    assert sympify(X) == X.as_immutable() == ImmutableSparseMatrix(2, 2, data)