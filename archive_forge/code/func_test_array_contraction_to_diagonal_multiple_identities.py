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
def test_array_contraction_to_diagonal_multiple_identities():
    expr = _array_contraction(_array_tensor_product(A, B, I, C), (1, 2, 4), (5, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(A, B, C), (1, 2, 4))
    expr = _array_contraction(_array_tensor_product(A, I, I), (1, 2, 4))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (A, [2])
    assert convert_array_to_matrix(expr) == A
    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 4), (3, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])
    expr = _array_contraction(_array_tensor_product(A, I, I, B), (1, 2, 3, 4, 6))
    assert _array_contraction_to_diagonal_multiple_identity(expr) == (expr, [])