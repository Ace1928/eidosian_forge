import random
from sympy import tensordiagonal, eye, KroneckerDelta, Array
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, ArraySymbol, ArrayElement, \
from sympy.testing.pytest import raises
def test_arrayexpr_array_shape():
    expr = _array_tensor_product(M, N, P, Q)
    assert expr.shape == (k, k, k, k, k, k, k, k)
    Z = MatrixSymbol('Z', m, n)
    expr = _array_tensor_product(M, Z)
    assert expr.shape == (k, k, m, n)
    expr2 = _array_contraction(expr, (0, 1))
    assert expr2.shape == (m, n)
    expr2 = _array_diagonal(expr, (0, 1))
    assert expr2.shape == (m, n, k)
    exprp = _permute_dims(expr, [2, 1, 3, 0])
    assert exprp.shape == (m, k, n, k)
    expr3 = _array_tensor_product(N, Z)
    expr2 = _array_add(expr, expr3)
    assert expr2.shape == (k, k, m, n)
    raises(ValueError, lambda: _array_contraction(expr, (1, 2)))
    raises(ValueError, lambda: _array_diagonal(expr, (1, 2)))
    raises(ValueError, lambda: _array_diagonal(expr, (1,)))