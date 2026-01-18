import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_inf_compare(self):
    aone = np.array(1)
    ainf = np.array(np.inf)
    self._assert_func(aone, ainf)
    self._assert_func(-ainf, aone)
    self._assert_func(-ainf, ainf)
    assert_raises(AssertionError, lambda: self._assert_func(ainf, aone))
    assert_raises(AssertionError, lambda: self._assert_func(aone, -ainf))
    assert_raises(AssertionError, lambda: self._assert_func(ainf, ainf))
    assert_raises(AssertionError, lambda: self._assert_func(ainf, -ainf))
    assert_raises(AssertionError, lambda: self._assert_func(-ainf, -ainf))