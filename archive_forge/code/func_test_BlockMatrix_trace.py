from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockMatrix_trace():
    A, B, C, D = [MatrixSymbol(s, 3, 3) for s in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    assert trace(X) == trace(A) + trace(D)
    assert trace(BlockMatrix([ZeroMatrix(n, n)])) == 0