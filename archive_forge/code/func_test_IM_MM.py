from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_IM_MM():
    assert isinstance(MM + IM, ImmutableMatrix)
    assert isinstance(IM + MM, ImmutableMatrix)
    assert isinstance(2 * IM + MM, ImmutableMatrix)
    assert MM.equals(IM)