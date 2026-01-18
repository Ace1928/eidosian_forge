import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_array_diffshape(self):
    """Test two arrays with different shapes are found not equal."""
    a = np.array([1, 2])
    b = np.array([[1, 2], [1, 2]])
    self._test_not_equal(a, b)