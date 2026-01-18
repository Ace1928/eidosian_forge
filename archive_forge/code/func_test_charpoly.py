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
def test_charpoly():
    x, y = (Symbol('x'), Symbol('y'))
    z, t = (Symbol('z'), Symbol('t'))
    from sympy.abc import a, b, c
    m = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert eye_Determinant(3).charpoly(x) == Poly((x - 1) ** 3, x)
    assert eye_Determinant(3).charpoly(y) == Poly((y - 1) ** 3, y)
    assert m.charpoly() == Poly(x ** 3 - 15 * x ** 2 - 18 * x, x)
    raises(NonSquareMatrixError, lambda: Matrix([[1], [2]]).charpoly())
    n = Matrix(4, 4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert n.charpoly() == Poly(x ** 4, x)
    n = Matrix(4, 4, [45, 0, 0, 0, 0, 23, 0, 0, 0, 0, 87, 0, 0, 0, 0, 12])
    assert n.charpoly() == Poly(x ** 4 - 167 * x ** 3 + 8811 * x ** 2 - 173457 * x + 1080540, x)
    n = Matrix(3, 3, [x, 0, 0, a, y, 0, b, c, z])
    assert n.charpoly() == Poly(t ** 3 - (x + y + z) * t ** 2 + t * (x * y + y * z + x * z) - x * y * z, t)