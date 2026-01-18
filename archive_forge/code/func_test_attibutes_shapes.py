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
@pytest.mark.parametrize('Est', (PLSSVD, PLSRegression, PLSCanonical))
def test_attibutes_shapes(Est):
    d = load_linnerud()
    X = d.data
    Y = d.target
    n_components = 2
    pls = Est(n_components=n_components)
    pls.fit(X, Y)
    assert all((attr.shape[1] == n_components for attr in (pls.x_weights_, pls.y_weights_)))