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
@XFAIL
def test_eigen_vects():
    m = Matrix(2, 2, [1, 0, 0, I])
    raises(NotImplementedError, lambda: m.is_diagonalizable(True))
    assert not m.is_diagonalizable(True)
    raises(MatrixError, lambda: m.diagonalize(True))
    P, D = m.diagonalize(True)