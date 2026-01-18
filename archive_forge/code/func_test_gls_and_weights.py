from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_gls_and_weights(self):
    assert_raises(ValueError, PCA, self.x, gls=True)
    assert_raises(ValueError, PCA, self.x, weights=np.array([1.0, 1.0]))
    x = self.x - self.x.mean(0)
    x = x / (x ** 2.0).mean(0)
    pc_gls = PCA(x, ncomp=1, standardize=False, demean=False, gls=True)
    pc = PCA(x, ncomp=1, standardize=False, demean=False)
    errors = x - pc.projection
    var = (errors ** 2.0).mean(0)
    weights = 1.0 / var
    weights = weights / np.sqrt((weights ** 2.0).mean())
    assert_allclose(weights, pc_gls.weights)
    assert_equal(x, pc_gls.data)
    assert_equal(x, pc.data)
    pc_weights = PCA(x, ncomp=1, standardize=False, demean=False, weights=weights)
    assert_allclose(weights, pc_weights.weights)
    assert_allclose(np.abs(pc_weights.factors), np.abs(pc_gls.factors))