import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_inf_compare_array(self):
    x = np.array([1.1, 2.2, np.inf])
    ainf = np.array(np.inf)
    assert_raises(AssertionError, lambda: self._assert_func(x, ainf))
    assert_raises(AssertionError, lambda: self._assert_func(ainf, x))
    assert_raises(AssertionError, lambda: self._assert_func(x, -ainf))
    assert_raises(AssertionError, lambda: self._assert_func(-x, -ainf))
    assert_raises(AssertionError, lambda: self._assert_func(-ainf, -x))
    self._assert_func(-ainf, x)