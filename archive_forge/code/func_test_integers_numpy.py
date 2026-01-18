import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_integers_numpy(self, dtype):
    high = np.array([1])
    low = np.array([0])
    out = self.rg.integers(low, high, dtype=dtype)
    assert out.shape == (1,)
    out = self.rg.integers(low[0], high, dtype=dtype)
    assert out.shape == (1,)
    out = self.rg.integers(low, high[0], dtype=dtype)
    assert out.shape == (1,)