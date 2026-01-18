from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
from sympy.printing.mathematica import mathematica_code as mcode
def test_NDArray():
    from sympy.tensor.array import MutableDenseNDimArray, ImmutableDenseNDimArray, MutableSparseNDimArray, ImmutableSparseNDimArray
    example = MutableDenseNDimArray([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    assert mcode(example) == '{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}'
    example = ImmutableDenseNDimArray(example)
    assert mcode(example) == '{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}'
    example = MutableSparseNDimArray(example)
    assert mcode(example) == 'SparseArray[{{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, {1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, {1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, {1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, {2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, {2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, {2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, {2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24}, {2, 3, 4}]'
    example = ImmutableSparseNDimArray(example)
    assert mcode(example) == 'SparseArray[{{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, {1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, {1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, {1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, {2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, {2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, {2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, {2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24}, {2, 3, 4}]'