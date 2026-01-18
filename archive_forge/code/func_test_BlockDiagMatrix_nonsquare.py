from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockDiagMatrix_nonsquare():
    A = MatrixSymbol('A', n, m)
    B = MatrixSymbol('B', k, l)
    X = BlockDiagMatrix(A, B)
    assert X.shape == (n + k, m + l)
    assert X.shape == (n + k, m + l)
    assert X.rowblocksizes == [n, k]
    assert X.colblocksizes == [m, l]
    C = MatrixSymbol('C', n, m)
    D = MatrixSymbol('D', k, l)
    Y = BlockDiagMatrix(C, D)
    assert block_collapse(X + Y) == BlockDiagMatrix(A + C, B + D)
    assert block_collapse(X * Y.T) == BlockDiagMatrix(A * C.T, B * D.T)
    raises(NonInvertibleMatrixError, lambda: BlockDiagMatrix(A, C.T).inverse())