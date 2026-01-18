import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_nanfunctions_matrices_general():
    mat = np.matrix(np.eye(3))
    for f in (np.nanargmin, np.nanargmax, np.nansum, np.nanprod, np.nanmean, np.nanvar, np.nanstd):
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))
    for f in (np.nancumsum, np.nancumprod):
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3 * 3))