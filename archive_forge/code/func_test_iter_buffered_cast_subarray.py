import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffered_cast_subarray():
    sdt1 = [('a', 'f4')]
    sdt2 = [('a', 'f8', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    for x, count in zip(i, list(range(6))):
        assert_(np.all(x['a'] == count))
    sdt1 = [('a', 'O', (1, 1))]
    sdt2 = [('a', 'O', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'], casting='unsafe', op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_(np.all(x['a'] == count))
            x['a'][0] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1) + 2)
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readwrite'], casting='unsafe', op_dtypes=sdt2)
    with i:
        assert_equal(i[0].dtype, np.dtype(sdt2))
        count = 0
        for x in i:
            assert_equal(x['a'], count)
            x['a'] += 2
            count += 1
    assert_equal(a['a'], np.arange(6).reshape(6, 1, 1, 1) * np.ones((1, 3, 2, 2)) + 2)
    sdt1 = [('a', 'f8', (3, 2, 2))]
    sdt2 = [('a', 'O', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (1,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'][:, 0, 0, 0] = np.arange(6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], count)
        count += 1
    sdt1 = [('a', 'O', (3, 2, 2))]
    sdt2 = [('a', 'f4', (3, 2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 3 * 2 * 2).reshape(6, 3, 2, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'])
        count += 1
    sdt1 = [('a', 'f8', (6,))]
    sdt2 = [('a', 'f4', (2,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 6).reshape(6, 6)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'], a[count]['a'][:2])
        count += 1
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (6,))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2], a[count]['a'])
        assert_equal(x['a'][2:], [0, 0, 0, 0])
        count += 1
    sdt1 = [('a', 'f8', (2,))]
    sdt2 = [('a', 'f4', (2, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][0], a[count]['a'])
        assert_equal(x['a'][1], a[count]['a'])
        count += 1
    sdt1 = [('a', 'f8', (2, 1))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2).reshape(6, 2, 1)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 0])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1
    sdt1 = [('a', 'f8', (2, 3))]
    sdt2 = [('a', 'f4', (3, 2))]
    a = np.zeros((6,), dtype=sdt1)
    a['a'] = np.arange(6 * 2 * 3).reshape(6, 2, 3)
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt2)
    assert_equal(i[0].dtype, np.dtype(sdt2))
    count = 0
    for x in i:
        assert_equal(x['a'][:2, 0], a[count]['a'][:, 0])
        assert_equal(x['a'][:2, 1], a[count]['a'][:, 1])
        assert_equal(x['a'][2, :], [0, 0])
        count += 1