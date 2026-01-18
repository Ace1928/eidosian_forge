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
def test_mul_index():
    assert (A * y)[0, 0] == A[0, 0] * y[0, 0] + A[0, 1] * y[1, 0]
    assert (A * B).as_mutable() == A.as_mutable() * B.as_mutable()
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    result = (X * Y)[4, 2]
    expected = Sum(X[4, i] * Y[i, 2], (i, 0, m - 1))
    assert result.args[0].dummy_eq(expected.args[0], i)
    assert result.args[1][1:] == expected.args[1][1:]