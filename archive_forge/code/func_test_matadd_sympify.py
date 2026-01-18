from sympy.matrices.expressions import MatrixSymbol, MatAdd, MatPow, MatMul
from sympy.matrices.expressions.special import GenericZeroMatrix, ZeroMatrix
from sympy.matrices.common import ShapeError
from sympy.matrices import eye, ImmutableMatrix
from sympy.core import Add, Basic, S
from sympy.core.add import add
from sympy.testing.pytest import XFAIL, raises
def test_matadd_sympify():
    assert isinstance(MatAdd(eye(1), eye(1)).args[0], Basic)
    assert isinstance(add(eye(1), eye(1)).args[0], Basic)