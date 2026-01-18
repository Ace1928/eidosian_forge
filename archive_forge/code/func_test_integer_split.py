import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_integer_split(self):
    a = np.arange(10)
    res = array_split(a, 1)
    desired = [np.arange(10)]
    compare_results(res, desired)
    res = array_split(a, 2)
    desired = [np.arange(5), np.arange(5, 10)]
    compare_results(res, desired)
    res = array_split(a, 3)
    desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
    compare_results(res, desired)
    res = array_split(a, 4)
    desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8), np.arange(8, 10)]
    compare_results(res, desired)
    res = array_split(a, 5)
    desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 10)]
    compare_results(res, desired)
    res = array_split(a, 6)
    desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
    compare_results(res, desired)
    res = array_split(a, 7)
    desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
    compare_results(res, desired)
    res = array_split(a, 8)
    desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
    compare_results(res, desired)
    res = array_split(a, 9)
    desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
    compare_results(res, desired)
    res = array_split(a, 10)
    desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
    compare_results(res, desired)
    res = array_split(a, 11)
    desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3), np.arange(3, 4), np.arange(4, 5), np.arange(5, 6), np.arange(6, 7), np.arange(7, 8), np.arange(8, 9), np.arange(9, 10), np.array([])]
    compare_results(res, desired)