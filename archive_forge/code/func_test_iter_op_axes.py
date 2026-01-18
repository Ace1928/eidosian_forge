import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_op_axes():
    a = arange(6).reshape(2, 3)
    i = nditer([a, a.T], [], [['readonly']] * 2, op_axes=[[0, 1], [1, 0]])
    assert_(all([x == y for x, y in i]))
    a = arange(24).reshape(2, 3, 4)
    i = nditer([a.T, a], [], [['readonly']] * 2, op_axes=[[2, 1, 0], None])
    assert_(all([x == y for x, y in i]))
    a = arange(1, 31).reshape(2, 3, 5)
    b = arange(1, 3)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [0, -1, -1]])
    assert_equal([x * y for x, y in i], (a * b.reshape(2, 1, 1)).ravel())
    b = arange(1, 4)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [-1, 0, -1]])
    assert_equal([x * y for x, y in i], (a * b.reshape(1, 3, 1)).ravel())
    b = arange(1, 6)
    i = nditer([a, b], [], [['readonly']] * 2, op_axes=[None, [np.newaxis, np.newaxis, 0]])
    assert_equal([x * y for x, y in i], (a * b.reshape(1, 1, 5)).ravel())
    a = arange(24).reshape(2, 3, 4)
    b = arange(40).reshape(5, 2, 4)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2, op_axes=[[0, 1, -1, -1], [-1, -1, 0, 1]])
    assert_equal(i.shape, (2, 3, 5, 2))
    a = arange(12).reshape(3, 4)
    b = arange(20).reshape(4, 5)
    i = nditer([a, b], ['multi_index'], [['readonly']] * 2, op_axes=[[0, -1], [-1, 1]])
    assert_equal(i.shape, (3, 5))