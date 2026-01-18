from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.common import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises
def test_matadd_of_matrices():
    assert MatAdd(eye(2), 4 * eye(2), eye(2)).doit() == ImmutableMatrix(6 * eye(2))
    assert add(eye(2), 4 * eye(2), eye(2)).doit() == ImmutableMatrix(6 * eye(2))