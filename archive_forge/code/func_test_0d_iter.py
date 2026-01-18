import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_0d_iter():
    i = nditer([2, 3], ['multi_index'], [['readonly']] * 2)
    assert_equal(i.ndim, 0)
    assert_equal(next(i), (2, 3))
    assert_equal(i.multi_index, ())
    assert_equal(i.iterindex, 0)
    assert_raises(StopIteration, next, i)
    i.reset()
    assert_equal(next(i), (2, 3))
    assert_raises(StopIteration, next, i)
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()])
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)
    i = nditer(np.arange(5), ['multi_index'], [['readonly']], op_axes=[()], itershape=())
    assert_equal(i.ndim, 0)
    assert_equal(len(i), 1)
    with assert_raises(ValueError):
        nditer(np.arange(5), ['multi_index'], [['readonly']], itershape=())
    sdt = [('a', 'f4'), ('b', 'i8'), ('c', 'c8', (2, 3)), ('d', 'O')]
    a = np.array(0.5, dtype='f4')
    i = nditer(a, ['buffered', 'refs_ok'], ['readonly'], casting='unsafe', op_dtypes=sdt)
    vals = next(i)
    assert_equal(vals['a'], 0.5)
    assert_equal(vals['b'], 0)
    assert_equal(vals['c'], [[0.5] * 3] * 2)
    assert_equal(vals['d'], 0.5)