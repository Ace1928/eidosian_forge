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
def test_diagonalize():
    m = Matrix(2, 2, [0, -1, 1, 0])
    raises(MatrixError, lambda: m.diagonalize(reals_only=True))
    P, D = m.diagonalize()
    assert D.is_diagonal()
    assert D == Matrix([[-I, 0], [0, I]])
    m = Matrix(2, 2, [0, 0.5, 0.5, 0])
    P, D = m.diagonalize()
    assert all((isinstance(e, Float) for e in D.values()))
    assert all((isinstance(e, Float) for e in P.values()))
    _, D2 = m.diagonalize(reals_only=True)
    assert D == D2
    m = Matrix([[0, 1, 0, 0], [1, 0, 0, 0.002], [0.002, 0, 0, 1], [0, 0, 1, 0]])
    P, D = m.diagonalize()
    assert allclose(P * D, m * P)