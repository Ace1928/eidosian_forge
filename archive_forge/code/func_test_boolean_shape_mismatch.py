import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_shape_mismatch(self):
    arr = np.ones((5, 4, 3))
    index = np.array([True])
    assert_raises(IndexError, arr.__getitem__, index)
    index = np.array([False] * 6)
    assert_raises(IndexError, arr.__getitem__, index)
    index = np.zeros((4, 4), dtype=bool)
    assert_raises(IndexError, arr.__getitem__, index)
    assert_raises(IndexError, arr.__getitem__, (slice(None), index))