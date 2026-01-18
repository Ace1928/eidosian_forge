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
def test_issue_20752():
    b = symbols('b', nonzero=True)
    m = Matrix([[0, 0, 0], [0, b, 0], [0, 0, b]])
    assert m.is_positive_semidefinite is None