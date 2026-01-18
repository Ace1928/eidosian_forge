from os import path
import warnings
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from scipy.io import readsav
from scipy.io import _idl
def test_arrays_replicated_3d(self):
    pth = path.join(DATA_PATH, 'struct_pointer_arrays_replicated_3d.sav')
    s = readsav(pth, verbose=False)
    assert_(s.arrays_rep.g.dtype.type is np.object_)
    assert_(s.arrays_rep.h.dtype.type is np.object_)
    assert_equal(s.arrays_rep.g.shape, (4, 3, 2))
    assert_equal(s.arrays_rep.h.shape, (4, 3, 2))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                assert_array_identical(s.arrays_rep.g[i, j, k], np.repeat(np.float32(4.0), 2).astype(np.object_))
                assert_array_identical(s.arrays_rep.h[i, j, k], np.repeat(np.float32(4.0), 3).astype(np.object_))
                g0 = vect_id(s.arrays_rep.g[i, j, k])
                g1 = id(s.arrays_rep.g[0, 0, 0][0])
                assert np.all(g0 == g1)
                h0 = vect_id(s.arrays_rep.h[i, j, k])
                h1 = id(s.arrays_rep.h[0, 0, 0][0])
                assert np.all(h0 == h1)