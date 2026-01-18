import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_empty_tuple_index(self):
    a = np.array([1, 2, 3])
    assert_equal(a[()], a)
    assert_(a[()].base is a)
    a = np.array(0)
    assert_(isinstance(a[()], np.int_))