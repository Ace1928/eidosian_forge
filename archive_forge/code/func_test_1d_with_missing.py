import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_1d_with_missing(self):
    x = self.data
    x[-1] = masked
    x -= x.mean()
    nx = x.compressed()
    assert_almost_equal(np.corrcoef(nx), corrcoef(x))
    assert_almost_equal(np.corrcoef(nx, rowvar=False), corrcoef(x, rowvar=False))
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'bias and ddof have no effect')
        assert_almost_equal(np.corrcoef(nx, rowvar=False, bias=True), corrcoef(x, rowvar=False, bias=True))
    try:
        corrcoef(x, allow_masked=False)
    except ValueError:
        pass
    nx = x[1:-1]
    assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1]))
    assert_almost_equal(np.corrcoef(nx, nx[::-1], rowvar=False), corrcoef(x, x[::-1], rowvar=False))
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'bias and ddof have no effect')
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1], bias=1))
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1], ddof=2))