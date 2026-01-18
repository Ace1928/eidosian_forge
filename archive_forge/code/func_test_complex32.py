from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_complex32(self):
    s = readsav(path.join(DATA_PATH, 'scalar_complex32.sav'), verbose=False)
    assert_identical(s.c32, np.complex64(31244420000000.0 - 2.312442e+31j))