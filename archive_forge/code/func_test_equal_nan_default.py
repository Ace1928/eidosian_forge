import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_equal_nan_default(self):
    a = np.array([np.nan])
    b = np.array([np.nan])
    assert_array_equal(a, b)
    assert_array_almost_equal(a, b)
    assert_array_less(a, b)
    assert_allclose(a, b)