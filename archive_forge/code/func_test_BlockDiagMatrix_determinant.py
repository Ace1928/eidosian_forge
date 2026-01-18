from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_BlockDiagMatrix_determinant():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', m, m)
    assert det(BlockDiagMatrix()) == 1
    assert det(BlockDiagMatrix(A)) == det(A)
    assert det(BlockDiagMatrix(A, B)) == det(A) * det(B)
    C = MatrixSymbol('C', m, n)
    D = MatrixSymbol('D', n, m)
    assert det(BlockDiagMatrix(C, D)) == 0