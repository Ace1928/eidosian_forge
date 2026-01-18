from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
@slow
def test_BlockMatrix_3x3_symbolic():
    rowblocksizes = (n, m, k)
    colblocksizes = (m, k, n)
    K = BlockMatrix([[MatrixSymbol('M%s%s' % (rows, cols), rows, cols) for cols in colblocksizes] for rows in rowblocksizes])
    collapse = block_collapse(K.I)
    assert isinstance(collapse, BlockMatrix)