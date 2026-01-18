import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_iterrange():
    buffersize = 5
    a = arange(24, dtype='i4').reshape(4, 3, 2)
    a_fort = a.ravel(order='F')
    i = nditer(a, ['ranged'], ['readonly'], order='F', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])
    i = nditer(a, ['ranged', 'buffered'], ['readonly'], order='F', op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal([x[()] for x in i], a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal([x[()] for x in i], a_fort[r[0]:r[1]])

    def get_array(i):
        val = np.array([], dtype='f8')
        for x in i:
            val = np.concatenate((val, x))
        return val
    i = nditer(a, ['ranged', 'buffered', 'external_loop'], ['readonly'], order='F', op_dtypes='f8', buffersize=buffersize)
    assert_equal(i.iterrange, (0, 24))
    assert_equal(get_array(i), a_fort)
    for r in [(0, 24), (1, 2), (3, 24), (5, 5), (0, 20), (23, 24)]:
        i.iterrange = r
        assert_equal(i.iterrange, r)
        assert_equal(get_array(i), a_fort[r[0]:r[1]])