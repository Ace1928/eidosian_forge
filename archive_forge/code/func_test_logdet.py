import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_logdet(self, reset_randomstate):
    d = 30
    dg = np.linspace(1, 2, d)
    root = np.random.normal(size=(d, 4))
    fac = FactoredPSDMatrix(dg, root)
    mat = fac.to_matrix()
    _, ld = np.linalg.slogdet(mat)
    ld2 = fac.logdet()
    assert_almost_equal(ld, ld2)