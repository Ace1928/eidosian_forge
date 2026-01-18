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
def test_arrayexpr_convert_array_to_matrix():
    cg = _array_contraction(_array_tensor_product(M), (0, 1))
    assert convert_array_to_matrix(cg) == Trace(M)
    cg = _array_contraction(_array_tensor_product(M, N), (0, 1), (2, 3))
    assert convert_array_to_matrix(cg) == Trace(M) * Trace(N)
    cg = _array_contraction(_array_tensor_product(M, N), (0, 3), (1, 2))
    assert convert_array_to_matrix(cg) == Trace(M * N)
    cg = _array_contraction(_array_tensor_product(M, N), (0, 2), (1, 3))
    assert convert_array_to_matrix(cg) == Trace(M * N.T)
    cg = convert_matrix_to_array(M * N * P)
    assert convert_array_to_matrix(cg) == M * N * P
    cg = convert_matrix_to_array(M * N.T * P)
    assert convert_array_to_matrix(cg) == M * N.T * P
    cg = _array_contraction(_array_tensor_product(M, N, P, Q), (1, 2), (5, 6))
    assert convert_array_to_matrix(cg) == _array_tensor_product(M * N, P * Q)
    cg = _array_contraction(_array_tensor_product(-2, M, N), (1, 2))
    assert convert_array_to_matrix(cg) == -2 * M * N
    a = MatrixSymbol('a', k, 1)
    b = MatrixSymbol('b', k, 1)
    c = MatrixSymbol('c', k, 1)
    cg = PermuteDims(_array_contraction(_array_tensor_product(a, ArrayAdd(_array_tensor_product(b, c), _array_tensor_product(c, b))), (2, 4)), [0, 1, 3, 2])
    assert convert_array_to_matrix(cg) == a * (b.T * c + c.T * b)
    za = ZeroArray(m, n)
    assert convert_array_to_matrix(za) == ZeroMatrix(m, n)
    cg = _array_tensor_product(3, M)
    assert convert_array_to_matrix(cg) == 3 * M
    expr = _array_contraction(_array_tensor_product(M, N, P, Q), (0, 2), (1, 4, 6))
    assert convert_array_to_matrix(expr) == _array_contraction(_array_tensor_product(M.T * N, P, Q), (0, 2, 4))
    x = MatrixSymbol('x', k, 1)
    cg = PermuteDims(_array_contraction(_array_tensor_product(OneArray(1), x, OneArray(1), DiagMatrix(Identity(1))), (0, 5)), Permutation(1, 2, 3))
    assert convert_array_to_matrix(cg) == x
    expr = ArrayAdd(M, PermuteDims(M, [1, 0]))
    assert convert_array_to_matrix(expr) == M + Transpose(M)