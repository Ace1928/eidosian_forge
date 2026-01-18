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
def test_arrayexpr_contraction_permutation_mix():
    Me = M.subs(k, 3).as_explicit()
    Ne = N.subs(k, 3).as_explicit()
    cg1 = _array_contraction(PermuteDims(_array_tensor_product(M, N), Permutation([0, 2, 1, 3])), (2, 3))
    cg2 = _array_contraction(_array_tensor_product(M, N), (1, 3))
    assert cg1 == cg2
    cge1 = tensorcontraction(permutedims(tensorproduct(Me, Ne), Permutation([0, 2, 1, 3])), (2, 3))
    cge2 = tensorcontraction(tensorproduct(Me, Ne), (1, 3))
    assert cge1 == cge2
    cg1 = _permute_dims(_array_tensor_product(M, N), Permutation([0, 1, 3, 2]))
    cg2 = _array_tensor_product(M, _permute_dims(N, Permutation([1, 0])))
    assert cg1 == cg2
    cg1 = _array_contraction(_permute_dims(_array_tensor_product(M, N, P, Q), Permutation([0, 2, 3, 1, 4, 5, 7, 6])), (1, 2), (3, 5))
    cg2 = _array_contraction(_array_tensor_product(M, N, P, _permute_dims(Q, Permutation([1, 0]))), (1, 5), (2, 3))
    assert cg1 == cg2
    cg1 = _array_contraction(_permute_dims(_array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 2, 7, 5, 3])), (0, 1), (2, 6), (3, 7))
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(M, P, Q, N), (0, 1), (2, 3), (4, 7)), [1, 0])
    assert cg1 == cg2
    cg1 = _array_contraction(_permute_dims(_array_tensor_product(M, N, P, Q), Permutation([1, 0, 4, 6, 7, 2, 5, 3])), (0, 1), (2, 6), (3, 7))
    cg2 = _permute_dims(_array_contraction(_array_tensor_product(_permute_dims(M, [1, 0]), N, P, Q), (0, 1), (3, 6), (4, 5)), Permutation([1, 0]))
    assert cg1 == cg2