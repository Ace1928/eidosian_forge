import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_boolean_indexing_list(self):
    a = np.array([1, 2, 3])
    b = [True, False, True]
    assert_equal(a[b], [1, 3])
    assert_equal(a[None, b], [[1, 3]])