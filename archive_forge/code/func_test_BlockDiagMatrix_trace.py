from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockDiagMatrix_trace():
    assert trace(BlockDiagMatrix()) == 0
    assert trace(BlockDiagMatrix(ZeroMatrix(n, n))) == 0
    A = MatrixSymbol('A', n, n)
    assert trace(BlockDiagMatrix(A)) == trace(A)
    B = MatrixSymbol('B', m, m)
    assert trace(BlockDiagMatrix(A, B)) == trace(A) + trace(B)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert isinstance(trace(BlockDiagMatrix(C, D)), Trace)