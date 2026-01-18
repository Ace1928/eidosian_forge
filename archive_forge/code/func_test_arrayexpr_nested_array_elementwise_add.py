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
def test_arrayexpr_nested_array_elementwise_add():
    cg = _array_contraction(_array_add(_array_tensor_product(M, N), _array_tensor_product(N, M)), (1, 2))
    result = _array_add(_array_contraction(_array_tensor_product(M, N), (1, 2)), _array_contraction(_array_tensor_product(N, M), (1, 2)))
    assert cg == result
    cg = _array_diagonal(_array_add(_array_tensor_product(M, N), _array_tensor_product(N, M)), (1, 2))
    result = _array_add(_array_diagonal(_array_tensor_product(M, N), (1, 2)), _array_diagonal(_array_tensor_product(N, M), (1, 2)))
    assert cg == result