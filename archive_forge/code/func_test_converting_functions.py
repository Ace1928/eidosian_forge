from copy import copy
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.core.containers import Dict
from sympy.core.function import diff
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices import SparseMatrix
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices import Matrix
from sympy.tensor.array.sparse_ndim_array import ImmutableSparseNDimArray
from sympy.testing.pytest import raises
def test_converting_functions():
    arr_list = [1, 2, 3, 4]
    arr_matrix = Matrix(((1, 2), (3, 4)))
    arr_ndim_array = ImmutableDenseNDimArray(arr_list, (2, 2))
    assert isinstance(arr_ndim_array, ImmutableDenseNDimArray)
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    arr_ndim_array = ImmutableDenseNDimArray(arr_matrix)
    assert isinstance(arr_ndim_array, ImmutableDenseNDimArray)
    assert arr_matrix.tolist() == arr_ndim_array.tolist()
    assert arr_matrix.shape == arr_ndim_array.shape