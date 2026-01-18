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
def test_convergence_fail():
    d = load_linnerud()
    X = d.data
    Y = d.target
    pls_nipals = PLSCanonical(n_components=X.shape[1], max_iter=2)
    with pytest.warns(ConvergenceWarning):
        pls_nipals.fit(X, Y)