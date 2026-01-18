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
def test_function_return_types():
    X = ImmutableMatrix([[1, 2], [3, 4]])
    Y = ImmutableMatrix([[1], [0]])
    q, r = X.QRdecomposition()
    assert (type(q), type(r)) == (ImmutableMatrix, ImmutableMatrix)
    assert type(X.LUsolve(Y)) == ImmutableMatrix
    assert type(X.QRsolve(Y)) == ImmutableMatrix
    X = ImmutableMatrix([[5, 2], [2, 7]])
    assert X.T == X
    assert X.is_symmetric
    assert type(X.cholesky()) == ImmutableMatrix
    L, D = X.LDLdecomposition()
    assert (type(L), type(D)) == (ImmutableMatrix, ImmutableMatrix)
    X = ImmutableMatrix([[1, 2], [2, 1]])
    assert X.is_diagonalizable()
    assert X.det() == -3
    assert X.norm(2) == 3
    assert type(X.eigenvects()[0][2][0]) == ImmutableMatrix
    assert type(zeros(3, 3).as_immutable().nullspace()[0]) == ImmutableMatrix
    X = ImmutableMatrix([[1, 0], [2, 1]])
    assert type(X.lower_triangular_solve(Y)) == ImmutableMatrix
    assert type(X.T.upper_triangular_solve(Y)) == ImmutableMatrix
    assert type(X.minor_submatrix(0, 0)) == ImmutableMatrix