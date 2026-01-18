from sympy import KroneckerProduct
from sympy.combinatorics import Permutation
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct, hadamard_product)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import OneMatrix
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import (Identity, ZeroMatrix)
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import hadamard_power
from sympy.tensor.array.expressions.array_expressions import ArrayAdd, ArrayTensorProduct, PermuteDims
def test_matrix_derivatives_of_traces():
    expr = Trace(A) * A
    I = Identity(k)
    assert expr.diff(A) == ArrayAdd(ArrayTensorProduct(I, A), PermuteDims(ArrayTensorProduct(Trace(A) * I, I), Permutation(3)(1, 2)))
    assert expr[i, j].diff(A[m, n]).doit() == KDelta(i, m) * KDelta(j, n) * Trace(A) + KDelta(m, n) * A[i, j]
    expr = Trace(X)
    assert expr.diff(X) == Identity(k)
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == KDelta(m, n)
    expr = Trace(X * A)
    assert expr.diff(X) == A.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit() == A[n, m]
    expr = Trace(A * X * B)
    assert expr.diff(X) == A.T * B.T
    assert expr.rewrite(Sum).diff(X[m, n]).doit().dummy_eq((A.T * B.T)[m, n])
    expr = Trace(A * X.T * B)
    assert expr.diff(X) == B * A
    expr = Trace(X.T * A)
    assert expr.diff(X) == A
    expr = Trace(A * X.T)
    assert expr.diff(X) == A
    expr = Trace(X ** 2)
    assert expr.diff(X) == 2 * X.T
    expr = Trace(X ** 2 * B)
    assert expr.diff(X) == (X * B + B * X).T
    expr = Trace(MatMul(X, X, B))
    assert expr.diff(X) == (X * B + B * X).T
    expr = Trace(X.T * B * X)
    assert expr.diff(X) == B * X + B.T * X
    expr = Trace(B * X * X.T)
    assert expr.diff(X) == B * X + B.T * X
    expr = Trace(X * X.T * B)
    assert expr.diff(X) == B * X + B.T * X
    expr = Trace(X * B * X.T)
    assert expr.diff(X) == X * B.T + X * B
    expr = Trace(B * X.T * X)
    assert expr.diff(X) == X * B.T + X * B
    expr = Trace(X.T * X * B)
    assert expr.diff(X) == X * B.T + X * B
    expr = Trace(A * X * B * X)
    assert expr.diff(X) == A.T * X.T * B.T + B.T * X.T * A.T
    expr = Trace(X.T * X)
    assert expr.diff(X) == 2 * X
    expr = Trace(X * X.T)
    assert expr.diff(X) == 2 * X
    expr = Trace(B.T * X.T * C * X * B)
    assert expr.diff(X) == C.T * X * B * B.T + C * X * B * B.T
    expr = Trace(X.T * B * X * C)
    assert expr.diff(X) == B * X * C + B.T * X * C.T
    expr = Trace(A * X * B * X.T * C)
    assert expr.diff(X) == A.T * C.T * X * B.T + C * A * X * B
    expr = Trace((A * X * B + C) * (A * X * B + C).T)
    assert expr.diff(X) == 2 * A.T * (A * X * B + C) * B.T
    expr = Trace(X ** k)
    expr = Trace(A * X ** k)
    expr = Trace(B.T * X.T * C * X * X.T * C * X * B)
    assert expr.diff(X) == C * X * X.T * C * X * B * B.T + C.T * X * B * B.T * X.T * C.T * X + C * X * B * B.T * X.T * C * X + C.T * X * X.T * C.T * X * B * B.T
    expr = Trace(A * X ** (-1) * B)
    assert expr.diff(X) == -Inverse(X).T * A.T * B.T * Inverse(X).T
    expr = Trace(Inverse(X.T * C * X) * A)
    assert expr.diff(X) == -X.inv().T * A.T * X.inv() * C.inv().T * X.inv().T - X.inv().T * A * X.inv() * C.inv() * X.inv().T
    expr = Trace((X.T * C * X).inv() * (X.T * B * X))
    assert expr.diff(X) == -2 * C * X * (X.T * C * X).inv() * X.T * B * X * (X.T * C * X).inv() + 2 * B * X * (X.T * C * X).inv()
    expr = Trace((A + X.T * C * X).inv() * (X.T * B * X))
    assert expr.diff(X) == B * X * Inverse(A + X.T * C * X) - C * X * Inverse(A + X.T * C * X) * X.T * B * X * Inverse(A + X.T * C * X) - C.T * X * Inverse(A.T + (C * X).T * X) * X.T * B.T * X * Inverse(A.T + (C * X).T * X) + B.T * X * Inverse(A.T + (C * X).T * X)