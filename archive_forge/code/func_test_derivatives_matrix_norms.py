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
def test_derivatives_matrix_norms():
    expr = x.T * y
    assert expr.diff(x) == y
    assert expr[0, 0].diff(x[m, 0]).doit() == y[m, 0]
    expr = (x.T * y) ** S.Half
    assert expr.diff(x) == y / (2 * sqrt(x.T * y))
    expr = (x.T * x) ** S.Half
    assert expr.diff(x) == x * (x.T * x) ** Rational(-1, 2)
    expr = (c.T * a * x.T * b) ** S.Half
    assert expr.diff(x) == b * a.T * c / sqrt(c.T * a * x.T * b) / 2
    expr = (c.T * a * x.T * b) ** Rational(1, 3)
    assert expr.diff(x) == b * a.T * c * (c.T * a * x.T * b) ** Rational(-2, 3) / 3
    expr = (a.T * X * b) ** S.Half
    assert expr.diff(X) == a / (2 * sqrt(a.T * X * b)) * b.T
    expr = d.T * x * (a.T * X * b) ** S.Half * y.T * c
    assert expr.diff(X) == a / (2 * sqrt(a.T * X * b)) * x.T * d * y.T * c * b.T