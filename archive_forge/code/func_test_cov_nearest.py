import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_cov_nearest(self):
    x = self.x
    res_r = self.res
    y = cov_nearest(x, method='nearest')
    assert_almost_equal(y, res_r.mat, decimal=3)
    d = norm_f(x, y)
    assert_allclose(d, res_r.normF, rtol=0.001)
    y = cov_nearest(x, method='clipped')
    assert_almost_equal(y, res_r.mat, decimal=2)
    d = norm_f(x, y)
    assert_allclose(d, res_r.normF, rtol=0.15)