from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_int32(self):
    s = readsav(path.join(DATA_PATH, 'scalar_int32.sav'), verbose=False)
    assert_identical(s.i32s, np.int32(-1234567890))