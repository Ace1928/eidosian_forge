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
def test_ndim_array_initiation():
    arr_with_no_elements = ImmutableDenseNDimArray([], shape=(0,))
    assert len(arr_with_no_elements) == 0
    assert arr_with_no_elements.rank() == 1
    raises(ValueError, lambda: ImmutableDenseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableDenseNDimArray([], shape=()))
    raises(ValueError, lambda: ImmutableSparseNDimArray([0], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([1, 2, 3], shape=(0,)))
    raises(ValueError, lambda: ImmutableSparseNDimArray([], shape=()))
    arr_with_one_element = ImmutableDenseNDimArray([23])
    assert len(arr_with_one_element) == 1
    assert arr_with_one_element[0] == 23
    assert arr_with_one_element[:] == ImmutableDenseNDimArray([23])
    assert arr_with_one_element.rank() == 1
    arr_with_symbol_element = ImmutableDenseNDimArray([Symbol('x')])
    assert len(arr_with_symbol_element) == 1
    assert arr_with_symbol_element[0] == Symbol('x')
    assert arr_with_symbol_element[:] == ImmutableDenseNDimArray([Symbol('x')])
    assert arr_with_symbol_element.rank() == 1
    number5 = 5
    vector = ImmutableDenseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector.rank() == 1
    vector = ImmutableSparseNDimArray.zeros(number5)
    assert len(vector) == number5
    assert vector.shape == (number5,)
    assert vector._sparse_array == Dict()
    assert vector.rank() == 1
    n_dim_array = ImmutableDenseNDimArray(range(3 ** 4), (3, 3, 3, 3))
    assert len(n_dim_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == (3, 3, 3, 3)
    assert n_dim_array.rank() == 4
    array_shape = (3, 3, 3, 3)
    sparse_array = ImmutableSparseNDimArray.zeros(*array_shape)
    assert len(sparse_array._sparse_array) == 0
    assert len(sparse_array) == 3 * 3 * 3 * 3
    assert n_dim_array.shape == array_shape
    assert n_dim_array.rank() == 4
    one_dim_array = ImmutableDenseNDimArray([2, 3, 1])
    assert len(one_dim_array) == 3
    assert one_dim_array.shape == (3,)
    assert one_dim_array.rank() == 1
    assert one_dim_array.tolist() == [2, 3, 1]
    shape = (3, 3)
    array_with_many_args = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_many_args) == 3 * 3
    assert array_with_many_args.shape == shape
    assert array_with_many_args[0, 0] == 0
    assert array_with_many_args.rank() == 2
    shape = (int(3), int(3))
    array_with_long_shape = ImmutableSparseNDimArray.zeros(*shape)
    assert len(array_with_long_shape) == 3 * 3
    assert array_with_long_shape.shape == shape
    assert array_with_long_shape[int(0), int(0)] == 0
    assert array_with_long_shape.rank() == 2
    vector_with_long_shape = ImmutableDenseNDimArray(range(5), int(5))
    assert len(vector_with_long_shape) == 5
    assert vector_with_long_shape.shape == (int(5),)
    assert vector_with_long_shape.rank() == 1
    raises(ValueError, lambda: vector_with_long_shape[int(5)])
    from sympy.abc import x
    for ArrayType in [ImmutableDenseNDimArray, ImmutableSparseNDimArray]:
        rank_zero_array = ArrayType(x)
        assert len(rank_zero_array) == 1
        assert rank_zero_array.shape == ()
        assert rank_zero_array.rank() == 0
        assert rank_zero_array[()] == x
        raises(ValueError, lambda: rank_zero_array[0])