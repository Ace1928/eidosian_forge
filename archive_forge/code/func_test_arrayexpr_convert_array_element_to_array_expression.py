from sympy import tanh
from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import IndexedBase
from sympy.combinatorics import Permutation
from sympy.tensor.array.expressions.array_expressions import ArrayContraction, ArrayTensorProduct, \
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array, _convert_indexed_to_array
from sympy.testing.pytest import raises
def test_arrayexpr_convert_array_element_to_array_expression():
    A = ArraySymbol('A', (k,))
    B = ArraySymbol('B', (k,))
    s = Sum(A[i] * B[i], (i, 0, k - 1))
    cg = convert_indexed_to_array(s)
    assert cg == ArrayContraction(ArrayTensorProduct(A, B), (0, 1))
    s = A[i] * B[i]
    cg = convert_indexed_to_array(s)
    assert cg == ArrayDiagonal(ArrayTensorProduct(A, B), (0, 1))
    s = A[i] * B[j]
    cg = convert_indexed_to_array(s, [i, j])
    assert cg == ArrayTensorProduct(A, B)
    cg = convert_indexed_to_array(s, [j, i])
    assert cg == ArrayTensorProduct(B, A)
    s = tanh(A[i] * B[j])
    cg = convert_indexed_to_array(s, [i, j])
    assert cg.dummy_eq(ArrayElementwiseApplyFunc(tanh, ArrayTensorProduct(A, B)))