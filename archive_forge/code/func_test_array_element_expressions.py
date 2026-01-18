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
def test_array_element_expressions():
    assert M[0, 0] * N[0, 0] == N[0, 0] * M[0, 0]
    assert M[0, 0].diff(M[0, 0]) == 1
    assert M[0, 0].diff(M[1, 0]) == 0
    assert M[0, 0].diff(N[0, 0]) == 0
    assert M[0, 1].diff(M[i, j]) == KroneckerDelta(i, 0) * KroneckerDelta(j, 1)
    assert M[0, 1].diff(N[i, j]) == 0
    K4 = ArraySymbol('K4', shape=(k, k, k, k))
    assert K4[i, j, k, l].diff(K4[1, 2, 3, 4]) == KroneckerDelta(i, 1) * KroneckerDelta(j, 2) * KroneckerDelta(k, 3) * KroneckerDelta(l, 4)