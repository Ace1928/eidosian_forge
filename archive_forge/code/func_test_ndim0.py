import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_ndim0(self):
    x = np.array(1.0)
    y = np.from_dlpack(x)
    assert_array_equal(x, y)