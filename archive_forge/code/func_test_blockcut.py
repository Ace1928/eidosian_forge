from sympy.matrices.expressions.trace import Trace
from sympy.testing.pytest import raises, slow
from sympy.matrices.expressions.blockmatrix import (
from sympy.matrices.expressions import (MatrixSymbol, Identity,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import (
from sympy.core import Tuple, symbols, Expr, S
from sympy.functions import transpose, im, re
def test_blockcut():
    A = MatrixSymbol('A', n, m)
    B = blockcut(A, (n / 2, n / 2), (m / 2, m / 2))
    assert B == BlockMatrix([[A[:n / 2, :m / 2], A[:n / 2, m / 2:]], [A[n / 2:, :m / 2], A[n / 2:, m / 2:]]])
    M = ImmutableMatrix(4, 4, range(16))
    B = blockcut(M, (2, 2), (2, 2))
    assert M == ImmutableMatrix(B)
    B = blockcut(M, (1, 3), (2, 2))
    assert ImmutableMatrix(B.blocks[0, 1]) == ImmutableMatrix([[2, 3]])