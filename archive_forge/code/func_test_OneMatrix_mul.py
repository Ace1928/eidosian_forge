from sympy.core.add import Add
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.core.relational import Eq
from sympy.concrete.summations import Sum
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import (
from sympy.matrices.expressions.matmul import MatMul
from sympy.testing.pytest import raises
def test_OneMatrix_mul():
    n, m, k = symbols('n m k', integer=True)
    w = MatrixSymbol('w', n, 1)
    assert OneMatrix(n, m) * OneMatrix(m, k) == OneMatrix(n, k) * m
    assert w * OneMatrix(1, 1) == w
    assert OneMatrix(1, 1) * w.T == w.T