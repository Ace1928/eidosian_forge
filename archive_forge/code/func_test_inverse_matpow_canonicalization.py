from sympy.core import symbols, S
from sympy.matrices.expressions import MatrixSymbol, Inverse, MatPow, ZeroMatrix, OneMatrix
from sympy.matrices.common import NonInvertibleMatrixError, NonSquareMatrixError
from sympy.matrices import eye, Identity
from sympy.testing.pytest import raises
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
def test_inverse_matpow_canonicalization():
    A = MatrixSymbol('A', 3, 3)
    assert Inverse(MatPow(A, 3)).doit() == MatPow(Inverse(A), 3).doit()