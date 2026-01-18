import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_index_split_high_bound(self):
    a = np.arange(10)
    indices = [0, 5, 7, 10, 12]
    res = array_split(a, indices, axis=-1)
    desired = [np.array([]), np.arange(0, 5), np.arange(5, 7), np.arange(7, 10), np.array([]), np.array([])]
    compare_results(res, desired)