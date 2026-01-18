import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_non_integer_argument_errors(self):
    a = np.array([[5]])
    assert_raises(TypeError, np.reshape, a, (1.0, 1.0, -1))
    assert_raises(TypeError, np.reshape, a, (np.array(1.0), -1))
    assert_raises(TypeError, np.take, a, [0], 1.0)
    assert_raises(TypeError, np.take, a, [0], np.float64(1.0))