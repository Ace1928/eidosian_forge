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
def test_identity_power():
    k = Identity(n)
    assert MatPow(k, 4).doit() == k
    assert MatPow(k, n).doit() == k
    assert MatPow(k, -3).doit() == k
    assert MatPow(k, 0).doit() == k
    l = Identity(3)
    assert MatPow(l, n).doit() == l
    assert MatPow(l, -1).doit() == l
    assert MatPow(l, 0).doit() == l