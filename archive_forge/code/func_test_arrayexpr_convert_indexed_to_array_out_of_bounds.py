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
def test_arrayexpr_convert_indexed_to_array_out_of_bounds():
    expr = Sum(M[i, i], (i, 0, 4))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, i], (i, 0, k))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, i], (i, 1, k - 1))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, j] * N[j, m], (j, 0, 4))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, j] * N[j, m], (j, 0, k))
    raises(ValueError, lambda: convert_indexed_to_array(expr))
    expr = Sum(M[i, j] * N[j, m], (j, 1, k - 1))
    raises(ValueError, lambda: convert_indexed_to_array(expr))