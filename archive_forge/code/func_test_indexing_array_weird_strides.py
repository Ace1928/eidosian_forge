import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_indexing_array_weird_strides(self):
    x = np.ones(10)
    x2 = np.ones((10, 2))
    ind = np.arange(10)[:, None, None, None]
    ind = np.broadcast_to(ind, (10, 55, 4, 4))
    assert_array_equal(x[ind], x[ind.copy()])
    zind = np.zeros(4, dtype=np.intp)
    assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])