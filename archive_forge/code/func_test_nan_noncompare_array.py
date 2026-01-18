import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_nan_noncompare_array(self):
    x = np.array([1.1, 2.2, 3.3])
    anan = np.array(np.nan)
    assert_raises(AssertionError, lambda: self._assert_func(x, anan))
    assert_raises(AssertionError, lambda: self._assert_func(anan, x))
    x = np.array([1.1, 2.2, np.nan])
    assert_raises(AssertionError, lambda: self._assert_func(x, anan))
    assert_raises(AssertionError, lambda: self._assert_func(anan, x))
    y = np.array([1.0, 2.0, np.nan])
    self._assert_func(y, x)
    assert_raises(AssertionError, lambda: self._assert_func(x, y))