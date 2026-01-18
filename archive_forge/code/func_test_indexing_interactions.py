from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_indexing_interactions():
    assert (a * IM)[1, 1] == 5 * a
    assert (SM + IM)[1, 1] == SM[1, 1] + IM[1, 1]
    assert (SM * IM)[1, 1] == SM[1, 0] * IM[0, 1] + SM[1, 1] * IM[1, 1] + SM[1, 2] * IM[2, 1]