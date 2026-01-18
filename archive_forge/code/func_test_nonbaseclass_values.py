import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_nonbaseclass_values(self):

    class SubClass(np.ndarray):

        def __array_finalize__(self, old):
            self.fill(99)
    a = np.zeros((5, 5))
    s = a.copy().view(type=SubClass)
    s.fill(1)
    a[[0, 1, 2, 3, 4], :] = s
    assert_((a == 1).all())
    a[:, [0, 1, 2, 3, 4]] = s
    assert_((a == 1).all())
    a.fill(0)
    a[...] = s
    assert_((a == 1).all())