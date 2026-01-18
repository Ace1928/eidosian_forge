from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_float64(self):
    s = readsav(path.join(DATA_PATH, 'scalar_float64.sav'), verbose=False)
    assert_identical(s.f64, np.float64(-1.1976931348623156e+307))