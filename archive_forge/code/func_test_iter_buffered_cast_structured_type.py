import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffered_cast_structured_type():
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.arange(3, dtype='f4') + 0.5
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = [np.array(x) for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[0.5] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[1.5] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.zeros((3,), dtype='O')
    a[0] = (0.5, 0.5, [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], 0.5)
    a[1] = (1.5, 1.5, [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]], 1.5)
    a[2] = (2.5, 2.5, [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], 2.5)
    if HAS_REFCOUNT:
        rc = sys.getrefcount(a[0])
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = [x.copy() for x in i]
    assert_equal(vals[0]['a'], 0.5)
    assert_equal(vals[0]['b'], 0)
    assert_equal(vals[0]['c'], [[0.5] * 3] * 2)
    assert_equal(vals[0]['d'], 0.5)
    assert_equal(vals[1]['a'], 1.5)
    assert_equal(vals[1]['b'], 1)
    assert_equal(vals[1]['c'], [[1.5] * 3] * 2)
    assert_equal(vals[1]['d'], 1.5)
    assert_equal(vals[0].dtype, np.dtype(sdt))
    vals, i, x = [None] * 3
    if HAS_REFCOUNT:
        assert_equal(sys.getrefcount(a[0]), rc)
    sdt = [('a', 'f4')]
    a = np.array([(5.5,), (8,)], dtype=sdt)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes='i4')
    assert_equal([x_[()] for x_ in i], [5, 8])
    sdt = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    a = np.array([(5.5, 7, 'test'), (8, 10, 11)], dtype=sdt)
    assert_raises(TypeError, lambda: nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes='i4'))
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('d', 'u2'), ('a', 'O'), ('b', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    assert_equal([np.array(x_) for x_ in i], [np.array((1, 2, 3), dtype=sdt2), np.array((4, 5, 6), dtype=sdt2)])