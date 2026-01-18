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
def test_array_expr_as_explicit_with_explicit_component_arrays():
    from sympy.abc import x, y, z, t
    A = Array([[x, y], [z, t]])
    assert ArrayTensorProduct(A, A).as_explicit() == tensorproduct(A, A)
    assert ArrayDiagonal(A, (0, 1)).as_explicit() == tensordiagonal(A, (0, 1))
    assert ArrayContraction(A, (0, 1)).as_explicit() == tensorcontraction(A, (0, 1))
    assert ArrayAdd(A, A).as_explicit() == A + A
    assert ArrayElementwiseApplyFunc(sin, A).as_explicit() == A.applyfunc(sin)
    assert PermuteDims(A, [1, 0]).as_explicit() == permutedims(A, [1, 0])
    assert Reshape(A, [4]).as_explicit() == A.reshape(4)