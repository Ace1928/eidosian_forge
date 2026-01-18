import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
def test_concatenate_axis_None(self):
    a = np.arange(4, dtype=np.float64).reshape((2, 2))
    b = list(range(3))
    c = ['x']
    r = np.concatenate((a, a), axis=None)
    assert_equal(r.dtype, a.dtype)
    assert_equal(r.ndim, 1)
    r = np.concatenate((a, b), axis=None)
    assert_equal(r.size, a.size + len(b))
    assert_equal(r.dtype, a.dtype)
    r = np.concatenate((a, b, c), axis=None, dtype='U')
    d = array(['0.0', '1.0', '2.0', '3.0', '0', '1', '2', 'x'])
    assert_array_equal(r, d)
    out = np.zeros(a.size + len(b))
    r = np.concatenate((a, b), axis=None)
    rout = np.concatenate((a, b), axis=None, out=out)
    assert_(out is rout)
    assert_equal(r, rout)