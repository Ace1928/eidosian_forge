import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_indexing_array_negative_strides(self):
    arro = np.zeros((4, 4))
    arr = arro[::-1, ::-1]
    slices = (slice(None), [0, 1, 2, 3])
    arr[slices] = 10
    assert_array_equal(arr, 10.0)