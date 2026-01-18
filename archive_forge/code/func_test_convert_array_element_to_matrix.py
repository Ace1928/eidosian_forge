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
def test_convert_array_element_to_matrix():
    expr = ArrayElement(M, (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M, i, j)
    expr = ArrayElement(_array_contraction(_array_tensor_product(M, N), (1, 3)), (i, j))
    assert convert_array_to_matrix(expr) == MatrixElement(M * N.T, i, j)
    expr = ArrayElement(_array_tensor_product(M, N), (i, j, m, n))
    assert convert_array_to_matrix(expr) == expr