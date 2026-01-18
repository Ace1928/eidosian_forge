import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_best_order_c_index_2d():
    a = arange(6)
    i = nditer(a.reshape(2, 3), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 1, 2, 3, 4, 5])
    i = nditer(a.reshape(2, 3).copy(order='F'), ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [0, 3, 1, 4, 2, 5])
    i = nditer(a.reshape(2, 3)[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 4, 5, 0, 1, 2])
    i = nditer(a.reshape(2, 3)[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 1, 0, 5, 4, 3])
    i = nditer(a.reshape(2, 3)[::-1, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 4, 3, 2, 1, 0])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [3, 0, 4, 1, 5, 2])
    i = nditer(a.reshape(2, 3).copy(order='F')[:, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [2, 5, 1, 4, 0, 3])
    i = nditer(a.reshape(2, 3).copy(order='F')[::-1, ::-1], ['c_index'], [['readonly']])
    assert_equal(iter_indices(i), [5, 2, 4, 1, 3, 0])