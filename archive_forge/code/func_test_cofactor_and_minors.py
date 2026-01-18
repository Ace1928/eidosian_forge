import random
from sympy.core.numbers import I
from sympy.core.numbers import Rational
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix, eye, ones
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
from sympy.matrices.common import NonSquareMatrixError
from sympy.functions.combinatorial.factorials import factorial, subfactorial
def test_cofactor_and_minors():
    x = Symbol('x')
    e = Matrix(4, 4, [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])
    m = Matrix([[x, 1, 3], [2, 9, 11], [12, 13, 14]])
    cm = Matrix([[4, 76, -122, 48], [-8, -14 * x - 68, 17 * x + 142, -4 * x - 72], [4, 14 * x - 8, -21 * x + 4, 8 * x], [0, -4 * x + 24, 8 * x - 48, -4 * x + 24]])
    sub = Matrix([[x, 1, 2], [4, 5, 6], [2, 9, 10]])
    assert e.minor_submatrix(1, 2) == m
    assert e.minor_submatrix(-1, -1) == sub
    assert e.minor(1, 2) == -17 * x - 142
    assert e.cofactor(1, 2) == 17 * x + 142
    assert e.cofactor_matrix() == cm
    assert e.cofactor_matrix(method='bareiss') == cm
    assert e.cofactor_matrix(method='berkowitz') == cm
    raises(ValueError, lambda: e.cofactor(4, 5))
    raises(ValueError, lambda: e.minor(4, 5))
    raises(ValueError, lambda: e.minor_submatrix(4, 5))
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    assert a.minor_submatrix(0, 0) == Matrix([[5, 6]])
    raises(ValueError, lambda: Matrix(0, 0, []).minor_submatrix(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor(0, 0))
    raises(NonSquareMatrixError, lambda: a.minor(0, 0))
    raises(NonSquareMatrixError, lambda: a.cofactor_matrix())