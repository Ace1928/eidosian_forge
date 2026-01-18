from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def test_block_index_symbolic():
    A1 = MatrixSymbol('A1', n, k)
    A2 = MatrixSymbol('A2', n, l)
    A3 = MatrixSymbol('A3', m, k)
    A4 = MatrixSymbol('A4', m, l)
    A = BlockMatrix([[A1, A2], [A3, A4]])
    assert A[0, 0] == MatrixElement(A, 0, 0)
    assert A[n - 1, k - 1] == A1[n - 1, k - 1]
    assert A[n, k] == A4[0, 0]
    assert A[n + m - 1, 0] == MatrixElement(A, n + m - 1, 0)
    assert A[0, k + l - 1] == MatrixElement(A, 0, k + l - 1)
    assert A[n + m - 1, k + l - 1] == MatrixElement(A, n + m - 1, k + l - 1)
    assert A[i, j] == MatrixElement(A, i, j)
    assert A[n + i, k + j] == MatrixElement(A, n + i, k + j)
    assert A[n - i - 1, k - j - 1] == MatrixElement(A, n - i - 1, k - j - 1)