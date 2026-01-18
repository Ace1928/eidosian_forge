from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_invalid_pointer():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s = readsav(path.join(DATA_PATH, 'invalid_pointer.sav'), verbose=False)
    assert_(len(w) == 1)
    assert_(str(w[0].message) == 'Variable referenced by pointer not found in heap: variable will be set to None')
    assert_identical(s['a'], np.array([None, None]))