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
def test_block_index_large():
    n, m, k = symbols('n m k', integer=True, positive=True)
    i = symbols('i', integer=True, nonnegative=True)
    A1 = MatrixSymbol('A1', n, n)
    A2 = MatrixSymbol('A2', n, m)
    A3 = MatrixSymbol('A3', n, k)
    A4 = MatrixSymbol('A4', m, n)
    A5 = MatrixSymbol('A5', m, m)
    A6 = MatrixSymbol('A6', m, k)
    A7 = MatrixSymbol('A7', k, n)
    A8 = MatrixSymbol('A8', k, m)
    A9 = MatrixSymbol('A9', k, k)
    A = BlockMatrix([[A1, A2, A3], [A4, A5, A6], [A7, A8, A9]])
    assert A[n + i, n + i] == MatrixElement(A, n + i, n + i)