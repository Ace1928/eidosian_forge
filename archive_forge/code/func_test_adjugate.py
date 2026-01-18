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
def test_adjugate():
    x = Symbol('x')
    e = Matrix(4, 4, [x, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 11, 12, 13, 14, 14])
    adj = Matrix([[4, -8, 4, 0], [76, -14 * x - 68, 14 * x - 8, -4 * x + 24], [-122, 17 * x + 142, -21 * x + 4, 8 * x - 48], [48, -4 * x - 72, 8 * x, -4 * x + 24]])
    assert e.adjugate() == adj
    assert e.adjugate(method='bareiss') == adj
    assert e.adjugate(method='berkowitz') == adj
    a = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
    raises(NonSquareMatrixError, lambda: a.adjugate())