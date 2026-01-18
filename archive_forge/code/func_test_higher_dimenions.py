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
def test_higher_dimenions():
    m3 = ImmutableDenseNDimArray(range(10, 34), (2, 3, 4))
    assert m3.tolist() == [[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]
    assert m3._get_tuple_index(0) == (0, 0, 0)
    assert m3._get_tuple_index(1) == (0, 0, 1)
    assert m3._get_tuple_index(4) == (0, 1, 0)
    assert m3._get_tuple_index(12) == (1, 0, 0)
    assert str(m3) == '[[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]]'
    m3_rebuilt = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]])
    assert m3 == m3_rebuilt
    m3_other = ImmutableDenseNDimArray([[[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]], [[22, 23, 24, 25], [26, 27, 28, 29], [30, 31, 32, 33]]], (2, 3, 4))
    assert m3 == m3_other