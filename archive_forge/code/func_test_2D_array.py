import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_2D_array(self):
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    try:
        dsplit(a, 2)
        assert_(0)
    except ValueError:
        pass