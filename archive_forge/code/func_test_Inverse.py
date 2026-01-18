from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.powsimp import powsimp
from sympy.testing.pytest import raises
from sympy.core.expr import unchanged
from sympy.core import symbols, S
from sympy.matrices import Identity, MatrixSymbol, ImmutableMatrix, ZeroMatrix, OneMatrix, Matrix
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions import MatPow, MatAdd, MatMul
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixElement
def test_Inverse():
    assert Inverse(MatPow(C, 0)).doit() == Identity(n)
    assert Inverse(MatPow(C, 1)).doit() == Inverse(C)
    assert Inverse(MatPow(C, 2)).doit() == MatPow(C, -2)
    assert Inverse(MatPow(C, -1)).doit() == C
    assert MatPow(Inverse(C), 0).doit() == Identity(n)
    assert MatPow(Inverse(C), 1).doit() == Inverse(C)
    assert MatPow(Inverse(C), 2).doit() == MatPow(C, -2)
    assert MatPow(Inverse(C), -1).doit() == C