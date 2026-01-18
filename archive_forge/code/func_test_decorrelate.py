import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_decorrelate(self, reset_randomstate):
    d = 30
    dg = np.linspace(1, 2, d)
    root = np.random.normal(size=(d, 4))
    fac = FactoredPSDMatrix(dg, root)
    mat = fac.to_matrix()
    rmat = np.linalg.cholesky(mat)
    dcr = fac.decorrelate(rmat)
    idm = np.dot(dcr, dcr.T)
    assert_almost_equal(idm, np.eye(d))
    rhs = np.random.normal(size=(d, 5))
    mat2 = np.dot(rhs.T, np.linalg.solve(mat, rhs))
    mat3 = fac.decorrelate(rhs)
    mat3 = np.dot(mat3.T, mat3)
    assert_almost_equal(mat2, mat3)