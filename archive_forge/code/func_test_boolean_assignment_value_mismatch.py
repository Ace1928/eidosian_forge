import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_assignment_value_mismatch(self):
    a = np.arange(4)

    def f(a, v):
        a[a > -1] = v
    assert_raises(ValueError, f, a, [])
    assert_raises(ValueError, f, a, [1, 2, 3])
    assert_raises(ValueError, f, a[:1], [1, 2, 3])