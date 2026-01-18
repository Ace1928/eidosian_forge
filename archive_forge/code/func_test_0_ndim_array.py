import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_0_ndim_array(self):
    x = np.array(473963742225900817127911193656584771)
    y = np.array(18535119325151578301457182298393896)
    assert_raises(AssertionError, self._assert_func, x, y)
    y = x
    self._assert_func(x, y)
    x = np.array(43)
    y = np.array(10)
    assert_raises(AssertionError, self._assert_func, x, y)
    y = x
    self._assert_func(x, y)