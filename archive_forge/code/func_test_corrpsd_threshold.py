import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
@pytest.mark.parametrize('threshold', [0, 1e-15, 1e-10, 1e-06])
def test_corrpsd_threshold(threshold):
    x = np.array([[1, -0.9, -0.9], [-0.9, 1, -0.9], [-0.9, -0.9, 1]])
    y = corr_nearest(x, n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=1e-06, atol=1e-15)
    y = corr_clipped(x, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)
    y = cov_nearest(x, method='nearest', n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=1e-06, atol=1e-15)
    y = cov_nearest(x, n_fact=100, threshold=threshold)
    evals = np.linalg.eigvalsh(y)
    assert_allclose(evals[0], threshold, rtol=0.25, atol=1e-15)