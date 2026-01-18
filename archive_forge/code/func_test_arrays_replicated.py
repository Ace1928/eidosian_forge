from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_arrays_replicated(self):
    s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays_replicated.sav'), verbose=False)
    assert_(s.arrays_rep.g.dtype.type is np.object_)
    assert_(s.arrays_rep.h.dtype.type is np.object_)
    assert_equal(s.arrays_rep.g.shape, (5,))
    assert_equal(s.arrays_rep.h.shape, (5,))
    for i in range(5):
        assert_array_identical(s.arrays_rep.g[i], np.repeat(np.float32(4.0), 2).astype(np.object_))
        assert_array_identical(s.arrays_rep.h[i], np.repeat(np.float32(4.0), 3).astype(np.object_))
        assert_(np.all(vect_id(s.arrays_rep.g[i]) == id(s.arrays_rep.g[0][0])))
        assert_(np.all(vect_id(s.arrays_rep.h[i]) == id(s.arrays_rep.h[0][0])))