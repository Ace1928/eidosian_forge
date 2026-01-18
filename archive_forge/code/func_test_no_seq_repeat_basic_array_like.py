import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_no_seq_repeat_basic_array_like(self):

    class ArrayLike:

        def __init__(self, arr):
            self.arr = arr

        def __array__(self):
            return self.arr
    for arr_like in (ArrayLike(np.ones(3)), memoryview(np.ones(3))):
        assert_array_equal(arr_like * np.float32(3.0), np.full(3, 3.0))
        assert_array_equal(np.float32(3.0) * arr_like, np.full(3, 3.0))
        assert_array_equal(arr_like * np.int_(3), np.full(3, 3))
        assert_array_equal(np.int_(3) * arr_like, np.full(3, 3))