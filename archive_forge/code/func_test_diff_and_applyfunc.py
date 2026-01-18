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
def test_diff_and_applyfunc():
    from sympy.abc import x, y, z
    md = ImmutableDenseNDimArray([[x, y], [x * z, x * y * z]])
    assert md.diff(x) == ImmutableDenseNDimArray([[1, 0], [z, y * z]])
    assert diff(md, x) == ImmutableDenseNDimArray([[1, 0], [z, y * z]])
    sd = ImmutableSparseNDimArray(md)
    assert sd == ImmutableSparseNDimArray([x, y, x * z, x * y * z], (2, 2))
    assert sd.diff(x) == ImmutableSparseNDimArray([[1, 0], [z, y * z]])
    assert diff(sd, x) == ImmutableSparseNDimArray([[1, 0], [z, y * z]])
    mdn = md.applyfunc(lambda x: x * 3)
    assert mdn == ImmutableDenseNDimArray([[3 * x, 3 * y], [3 * x * z, 3 * x * y * z]])
    assert md != mdn
    sdn = sd.applyfunc(lambda x: x / 2)
    assert sdn == ImmutableSparseNDimArray([[x / 2, y / 2], [x * z / 2, x * y * z / 2]])
    assert sd != sdn
    sdp = sd.applyfunc(lambda x: x + 1)
    assert sdp == ImmutableSparseNDimArray([[x + 1, y + 1], [x * z + 1, x * y * z + 1]])
    assert sd != sdp