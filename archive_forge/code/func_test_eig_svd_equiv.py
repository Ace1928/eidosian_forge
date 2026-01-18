from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_eig_svd_equiv(self):
    pc_eig = PCA(self.x)
    pc_svd = PCA(self.x, method='svd')
    assert_allclose(pc_eig.projection, pc_svd.projection)
    assert_allclose(np.abs(pc_eig.factors[:, :2]), np.abs(pc_svd.factors[:, :2]))
    assert_allclose(np.abs(pc_eig.coeff[:2, :]), np.abs(pc_svd.coeff[:2, :]))
    assert_allclose(pc_eig.eigenvals, pc_svd.eigenvals)
    assert_allclose(np.abs(pc_eig.eigenvecs[:, :2]), np.abs(pc_svd.eigenvecs[:, :2]))
    pc_svd = PCA(self.x, method='svd', ncomp=2)
    pc_nipals = PCA(self.x, method='nipals', ncomp=2)
    assert_allclose(np.abs(pc_nipals.factors), np.abs(pc_svd.factors), atol=DECIMAL_5)
    assert_allclose(np.abs(pc_nipals.coeff), np.abs(pc_svd.coeff), atol=DECIMAL_5)
    assert_allclose(pc_nipals.eigenvals, pc_svd.eigenvals, atol=DECIMAL_5)
    assert_allclose(np.abs(pc_nipals.eigenvecs), np.abs(pc_svd.eigenvecs), atol=DECIMAL_5)
    assert_equal(self.x, pc_svd.data)
    assert_equal(self.x, pc_eig.data)
    assert_equal(self.x, pc_nipals.data)