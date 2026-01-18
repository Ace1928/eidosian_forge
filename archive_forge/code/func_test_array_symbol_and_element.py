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
def test_array_symbol_and_element():
    A = ArraySymbol('A', (2,))
    A0 = ArrayElement(A, (0,))
    A1 = ArrayElement(A, (1,))
    assert A[0] == A0
    assert A[1] != A0
    assert A.as_explicit() == ImmutableDenseNDimArray([A0, A1])
    A2 = tensorproduct(A, A)
    assert A2.shape == (2, 2)
    A3 = tensorcontraction(A2, (0, 1))
    assert A3.shape == ()
    A = ArraySymbol('A', (2, 3, 4))
    Ae = A.as_explicit()
    assert Ae == ImmutableDenseNDimArray([[[ArrayElement(A, (i, j, k)) for k in range(4)] for j in range(3)] for i in range(2)])
    p = _permute_dims(A, Permutation(0, 2, 1))
    assert isinstance(p, PermuteDims)
    A = ArraySymbol('A', (2,))
    raises(IndexError, lambda: A[()])
    raises(IndexError, lambda: A[0, 1])
    raises(ValueError, lambda: A[-1])
    raises(ValueError, lambda: A[2])
    O = OneArray(3, 4)
    Z = ZeroArray(m, n)
    raises(IndexError, lambda: O[()])
    raises(IndexError, lambda: O[1, 2, 3])
    raises(ValueError, lambda: O[3, 0])
    raises(ValueError, lambda: O[0, 4])
    assert O[1, 2] == 1
    assert Z[1, 2] == 0