import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_clipped(self):
    x = self.x
    res_r = self.res
    y = corr_clipped(x, threshold=1e-07)
    assert_almost_equal(y, res_r.mat, decimal=1)
    d = norm_f(x, y)
    assert_allclose(d, res_r.normF, rtol=0.15)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals, res_r.eigenvalues[::-1], rtol=0.1, atol=1e-07)
    assert_allclose(evals[0], 1e-07, rtol=0.02)