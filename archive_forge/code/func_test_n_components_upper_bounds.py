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
@pytest.mark.parametrize('Estimator', (PLSSVD, PLSRegression, PLSCanonical, CCA))
def test_n_components_upper_bounds(Estimator):
    """Check the validation of `n_components` upper bounds for `PLS` regressors."""
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5)
    Y = rng.randn(10, 3)
    est = Estimator(n_components=10)
    err_msg = '`n_components` upper bound is .*. Got 10 instead. Reduce `n_components`.'
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, Y)