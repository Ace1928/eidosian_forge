from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_idict(self):
    custom_dict = {'a': np.int16(999)}
    original_id = id(custom_dict)
    s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'), idict=custom_dict, verbose=False)
    assert_equal(original_id, id(s))
    assert_('a' in s)
    assert_identical(s['a'], np.int16(999))
    assert_identical(s['i8u'], np.uint8(234))