import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from . import util
def test_ftype(self):
    ftype = self.module
    ftype.foo()
    assert_equal(ftype.data.a, 0)
    ftype.data.a = 3
    ftype.data.x = [1, 2, 3]
    assert_equal(ftype.data.a, 3)
    assert_array_equal(ftype.data.x, np.array([1, 2, 3], dtype=np.float32))
    ftype.data.x[1] = 45
    assert_array_equal(ftype.data.x, np.array([1, 45, 3], dtype=np.float32))