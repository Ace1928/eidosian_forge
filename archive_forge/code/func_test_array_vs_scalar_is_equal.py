import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_vs_scalar_is_equal(self):
    """Test comparing an array with a scalar when all values are equal."""
    a = np.array([1.0, 1.0, 1.0])
    b = 1.0
    self._test_equal(a, b)