import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_flip_axes(self):
    a = arange(12).reshape(2, 3, 2)[::-1, ::-1, ::-1]
    i, j = np.nested_iters(a, [[0], [1, 2]])
    vals = [list(j) for _ in i]
    assert_equal(vals, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    i, j = np.nested_iters(a, [[0, 1], [2]])
    vals = [list(j) for _ in i]
    assert_equal(vals, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    i, j = np.nested_iters(a, [[0, 2], [1]])
    vals = [list(j) for _ in i]
    assert_equal(vals, [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]])
    i, j = np.nested_iters(a, [[0], [1, 2]], order='C')
    vals = [list(j) for _ in i]
    assert_equal(vals, [[11, 10, 9, 8, 7, 6], [5, 4, 3, 2, 1, 0]])
    i, j = np.nested_iters(a, [[0, 1], [2]], order='C')
    vals = [list(j) for _ in i]
    assert_equal(vals, [[11, 10], [9, 8], [7, 6], [5, 4], [3, 2], [1, 0]])
    i, j = np.nested_iters(a, [[0, 2], [1]], order='C')
    vals = [list(j) for _ in i]
    assert_equal(vals, [[11, 9, 7], [10, 8, 6], [5, 3, 1], [4, 2, 0]])