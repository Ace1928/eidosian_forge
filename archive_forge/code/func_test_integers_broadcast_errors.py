import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
def test_integers_broadcast_errors(self, dtype):
    if dtype == np.bool_:
        upper = 2
        lower = 0
    else:
        info = np.iinfo(dtype)
        upper = int(info.max) + 1
        lower = info.min
    with pytest.raises(ValueError):
        self.rg.integers(lower, [upper + 1] * 10, dtype=dtype)
    with pytest.raises(ValueError):
        self.rg.integers(lower - 1, [upper] * 10, dtype=dtype)
    with pytest.raises(ValueError):
        self.rg.integers([lower - 1], [upper] * 10, dtype=dtype)
    with pytest.raises(ValueError):
        self.rg.integers([0], [0], dtype=dtype)