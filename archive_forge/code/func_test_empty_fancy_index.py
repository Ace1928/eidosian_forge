import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_empty_fancy_index(self):
    a = np.array([1, 2, 3])
    assert_equal(a[[]], [])
    assert_equal(a[[]].dtype, a.dtype)
    b = np.array([], dtype=np.intp)
    assert_equal(a[[]], [])
    assert_equal(a[[]].dtype, a.dtype)
    b = np.array([])
    assert_raises(IndexError, a.__getitem__, b)