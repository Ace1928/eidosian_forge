import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
def test_dlpack_device(self):
    x = np.arange(5)
    assert x.__dlpack_device__() == (1, 0)
    y = np.from_dlpack(x)
    assert y.__dlpack_device__() == (1, 0)
    z = y[::2]
    assert z.__dlpack_device__() == (1, 0)