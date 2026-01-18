from sympy.core.evalf import N
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices import eye, Matrix
from sympy.core.singleton import S
from sympy.testing.pytest import raises, XFAIL
from sympy.matrices.matrices import NonSquareMatrixError, MatrixError
from sympy.matrices.expressions.fourier import DFT
from sympy.simplify.simplify import simplify
from sympy.matrices.immutable import ImmutableMatrix
from sympy.testing.pytest import slow
from sympy.testing.matrices import allclose
def test_jordan_form():
    m = Matrix(3, 2, [-3, 1, -3, 20, 3, 10])
    raises(NonSquareMatrixError, lambda: m.jordan_form())
    m = Matrix(4, 4, [2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2])
    P, J = m.jordan_form()
    assert m == J
    m = Matrix(4, 4, [2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2])
    P, J = m.jordan_form()
    assert m == J
    A = Matrix([[2, 4, 1, 0], [-4, 2, 0, 1], [0, 0, 2, 4], [0, 0, -4, 2]])
    P, J = A.jordan_form()
    assert simplify(P * J * P.inv()) == A
    assert Matrix(1, 1, [1]).jordan_form() == (Matrix([1]), Matrix([1]))
    assert Matrix(1, 1, [1]).jordan_form(calc_transform=False) == Matrix([1])
    m = Matrix([[3, 0, 0, 0, -3], [0, -3, -3, 0, 3], [0, 3, 0, 3, 0], [0, 0, 3, 0, 3], [3, 0, 0, 3, 0]])
    raises(MatrixError, lambda: m.jordan_form())
    m = Matrix([[0.6875, 0.125 + 0.1875 * sqrt(3)], [0.125 + 0.1875 * sqrt(3), 0.3125]])
    P, J = m.jordan_form()
    assert all((isinstance(x, Float) or x == 0 for x in P))
    assert all((isinstance(x, Float) or x == 0 for x in J))