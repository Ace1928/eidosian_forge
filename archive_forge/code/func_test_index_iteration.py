from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_index_iteration():
    L = TensorIndexType('Lorentz', dummy_name='L')
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', L)
    L0 = tensor_indices('L_0', L)
    L1 = tensor_indices('L_1', L)
    A = TensorHead('A', [L, L])
    B = TensorHead('B', [L, L], TensorSymmetry.fully_symmetric(2))
    e1 = A(i0, i2)
    e2 = A(i0, -i0)
    e3 = A(i0, i1) * B(i2, i3)
    e4 = A(i0, i1) * B(i2, -i1)
    e5 = A(i0, i1) * B(-i0, -i1)
    e6 = e1 + e4
    assert list(e1._iterate_free_indices) == [(i0, (1, 0)), (i2, (1, 1))]
    assert list(e1._iterate_dummy_indices) == []
    assert list(e1._iterate_indices) == [(i0, (1, 0)), (i2, (1, 1))]
    assert list(e2._iterate_free_indices) == []
    assert list(e2._iterate_dummy_indices) == [(L0, (1, 0)), (-L0, (1, 1))]
    assert list(e2._iterate_indices) == [(L0, (1, 0)), (-L0, (1, 1))]
    assert list(e3._iterate_free_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]
    assert list(e3._iterate_dummy_indices) == []
    assert list(e3._iterate_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]
    assert list(e4._iterate_free_indices) == [(i0, (0, 1, 0)), (i2, (1, 1, 0))]
    assert list(e4._iterate_dummy_indices) == [(L0, (0, 1, 1)), (-L0, (1, 1, 1))]
    assert list(e4._iterate_indices) == [(i0, (0, 1, 0)), (L0, (0, 1, 1)), (i2, (1, 1, 0)), (-L0, (1, 1, 1))]
    assert list(e5._iterate_free_indices) == []
    assert list(e5._iterate_dummy_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]
    assert list(e5._iterate_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]
    assert list(e6._iterate_free_indices) == [(i0, (0, 0, 1, 0)), (i2, (0, 1, 1, 0)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]
    assert list(e6._iterate_dummy_indices) == [(L0, (0, 0, 1, 1)), (-L0, (0, 1, 1, 1))]
    assert list(e6._iterate_indices) == [(i0, (0, 0, 1, 0)), (L0, (0, 0, 1, 1)), (i2, (0, 1, 1, 0)), (-L0, (0, 1, 1, 1)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]
    assert e1.get_indices() == [i0, i2]
    assert e1.get_free_indices() == [i0, i2]
    assert e2.get_indices() == [L0, -L0]
    assert e2.get_free_indices() == []
    assert e3.get_indices() == [i0, i1, i2, i3]
    assert e3.get_free_indices() == [i0, i1, i2, i3]
    assert e4.get_indices() == [i0, L0, i2, -L0]
    assert e4.get_free_indices() == [i0, i2]
    assert e5.get_indices() == [L0, L1, -L0, -L1]
    assert e5.get_free_indices() == []