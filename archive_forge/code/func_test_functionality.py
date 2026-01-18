import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_functionality(self):
    s = (2, 3, 4, 5)
    a = np.empty(s)
    for axis in range(-5, 4):
        b = expand_dims(a, axis)
        assert_(b.shape[axis] == 1)
        assert_(np.squeeze(b).shape == s)