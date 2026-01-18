from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_complex64(self):
    s = readsav(path.join(DATA_PATH, 'scalar_complex64.sav'), verbose=False)
    assert_identical(s.c64, np.complex128(1.1987253647623157e+112 - 5.198725888772916e+307j))