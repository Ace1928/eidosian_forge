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
def test_matrix_derivative_by_scalar():
    assert A.diff(i) == ZeroMatrix(k, k)
    assert (A * (X + B) * c).diff(i) == ZeroMatrix(k, 1)
    assert x.diff(i) == ZeroMatrix(k, 1)
    assert (x.T * y).diff(i) == ZeroMatrix(1, 1)
    assert (x * x.T).diff(i) == ZeroMatrix(k, k)
    assert (x + y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, 2).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_power(x, i).diff(i).dummy_eq(HadamardProduct(x.applyfunc(log), HadamardPower(x, i)))
    assert hadamard_product(x, y).diff(i) == ZeroMatrix(k, 1)
    assert hadamard_product(i * OneMatrix(k, 1), x, y).diff(i) == hadamard_product(x, y)
    assert (i * x).diff(i) == x
    assert (sin(i) * A * B * x).diff(i) == cos(i) * A * B * x
    assert x.applyfunc(sin).diff(i) == ZeroMatrix(k, 1)
    assert Trace(i ** 2 * X).diff(i) == 2 * i * Trace(X)
    mu = symbols('mu')
    expr = 2 * mu * x
    assert expr.diff(x) == 2 * mu * Identity(k)