import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_axis_tuple(self):
    a = np.empty((3, 3, 3))
    assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
    assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
    assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
    assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)