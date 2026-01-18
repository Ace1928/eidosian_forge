import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_nan_array(self):
    anan = np.array(np.nan)
    aone = np.array(1)
    ainf = np.array(np.inf)
    self._assert_func(anan, anan)
    assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
    assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
    assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))