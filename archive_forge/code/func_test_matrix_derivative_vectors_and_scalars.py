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
def test_matrix_derivative_vectors_and_scalars():
    assert x.diff(x) == Identity(k)
    assert x[i, 0].diff(x[m, 0]).doit() == KDelta(m, i)
    assert x.T.diff(x) == Identity(k)
    expr = x.T * a
    assert expr.diff(x) == a
    assert expr[0, 0].diff(x[m, 0]).doit() == a[m, 0]
    expr = a.T * x
    assert expr.diff(x) == a
    expr = a.T * X * b
    assert expr.diff(X) == a * b.T
    expr = a.T * X.T * b
    assert expr.diff(X) == b * a.T
    expr = a.T * X * a
    assert expr.diff(X) == a * a.T
    expr = a.T * X.T * a
    assert expr.diff(X) == a * a.T
    expr = b.T * X.T * X * c
    assert expr.diff(X) == X * b * c.T + X * c * b.T
    expr = (B * x + b).T * C * (D * x + d)
    assert expr.diff(x) == B.T * C * (D * x + d) + D.T * C.T * (B * x + b)
    expr = x.T * B * x
    assert expr.diff(x) == B * x + B.T * x
    expr = b.T * X.T * D * X * c
    assert expr.diff(X) == D.T * X * b * c.T + D * X * c * b.T
    expr = (X * b + c).T * D * (X * b + c)
    assert expr.diff(X) == D * (X * b + c) * b.T + D.T * (X * b + c) * b.T
    assert str(expr[0, 0].diff(X[m, n]).doit()) == 'b[n, 0]*Sum((c[_i_1, 0] + Sum(X[_i_1, _i_3]*b[_i_3, 0], (_i_3, 0, k - 1)))*D[_i_1, m], (_i_1, 0, k - 1)) + Sum((c[_i_2, 0] + Sum(X[_i_2, _i_4]*b[_i_4, 0], (_i_4, 0, k - 1)))*D[m, _i_2]*b[n, 0], (_i_2, 0, k - 1))'
    expr = x * x.T * x
    I = Identity(k)
    assert expr.diff(x) == KroneckerProduct(I, x.T * x) + 2 * x * x.T