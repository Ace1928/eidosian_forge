import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_nanfunctions_matrices():
    mat = np.matrix(np.eye(3))
    for f in [np.nanmin, np.nanmax]:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))
    mat[1] = np.nan
    for f in [np.nanmin, np.nanmax]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=0)
            assert_(isinstance(res, np.matrix))
            assert_(not np.any(np.isnan(res)))
            assert_(len(w) == 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=1)
            assert_(isinstance(res, np.matrix))
            assert_(np.isnan(res[1, 0]) and (not np.isnan(res[0, 0])) and (not np.isnan(res[2, 0])))
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat)
            assert_(np.isscalar(res))
            assert_(res != np.nan)
            assert_(len(w) == 0)