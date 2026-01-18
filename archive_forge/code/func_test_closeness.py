import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_closeness(self):
    self._assert_func(1.499999, 0.0, decimal=0)
    assert_raises(AssertionError, lambda: self._assert_func(1.5, 0.0, decimal=0))
    self._assert_func([1.499999], [0.0], decimal=0)
    assert_raises(AssertionError, lambda: self._assert_func([1.5], [0.0], decimal=0))