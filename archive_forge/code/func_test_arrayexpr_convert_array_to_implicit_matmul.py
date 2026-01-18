from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
from sympy.testing.pytest import raises
def test_arrayexpr_convert_array_to_implicit_matmul():
    cg = _array_tensor_product(a, b)
    assert convert_array_to_matrix(cg) == a * b.T
    cg = _array_tensor_product(a, b, I)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a * b.T, I)
    cg = _array_tensor_product(I, a, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(I, a * b.T)
    cg = _array_tensor_product(a, I, b)
    assert convert_array_to_matrix(cg) == _array_tensor_product(a, I, b)
    cg = _array_contraction(_array_tensor_product(I, I), (1, 2))
    assert convert_array_to_matrix(cg) == I
    cg = PermuteDims(_array_tensor_product(I, Identity(1)), [0, 2, 1, 3])
    assert convert_array_to_matrix(cg) == I