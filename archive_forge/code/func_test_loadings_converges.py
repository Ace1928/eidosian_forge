import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from sklearn.cross_decomposition._pls import (
from sklearn.datasets import load_linnerud, make_regression
from sklearn.ensemble import VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
def test_loadings_converges(global_random_seed):
    """Test that CCA converges. Non-regression test for #19549."""
    X, y = make_regression(n_samples=200, n_features=20, n_targets=20, random_state=global_random_seed)
    cca = CCA(n_components=10, max_iter=500)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        cca.fit(X, y)
    assert np.all(np.abs(cca.x_loadings_) < 1)