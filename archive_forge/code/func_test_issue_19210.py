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
def test_issue_19210():
    t = Symbol('t')
    H = Matrix([[3, 0, 0, 0], [0, 1, 2, 0], [0, 2, 2, 0], [0, 0, 0, 4]])
    A = (-I * H * t).jordan_form()
    assert A == (Matrix([[0, 1, 0, 0], [0, 0, -4 / (-1 + sqrt(17)), 4 / (1 + sqrt(17))], [0, 0, 1, 1], [1, 0, 0, 0]]), Matrix([[-4 * I * t, 0, 0, 0], [0, -3 * I * t, 0, 0], [0, 0, t * (-3 * I / 2 + sqrt(17) * I / 2), 0], [0, 0, 0, t * (-sqrt(17) * I / 2 - 3 * I / 2)]]))