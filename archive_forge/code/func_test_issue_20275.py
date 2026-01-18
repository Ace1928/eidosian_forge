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
def test_issue_20275():
    A = DFT(3).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (-1, 1, [Matrix([[1 - sqrt(3)], [1], [1]])])
    assert eigenvects[1] == (1, 1, [Matrix([[1 + sqrt(3)], [1], [1]])])
    assert eigenvects[2] == (-I, 1, [Matrix([[0], [-1], [1]])])
    A = DFT(4).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (-1, 1, [Matrix([[-1], [1], [1], [1]])])
    assert eigenvects[1] == (1, 2, [Matrix([[1], [0], [1], [0]]), Matrix([[2], [1], [0], [1]])])
    assert eigenvects[2] == (-I, 1, [Matrix([[0], [-1], [0], [1]])])
    A = DFT(5).as_explicit().expand(complex=True)
    eigenvects = A.eigenvects()
    assert eigenvects[0] == (-1, 1, [Matrix([[1 - sqrt(5)], [1], [1], [1], [1]])])
    assert eigenvects[1] == (1, 2, [Matrix([[S(1) / 2 + sqrt(5) / 2], [0], [1], [1], [0]]), Matrix([[S(1) / 2 + sqrt(5) / 2], [1], [0], [0], [1]])])