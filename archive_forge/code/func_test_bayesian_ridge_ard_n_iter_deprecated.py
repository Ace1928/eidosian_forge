from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
@pytest.mark.parametrize('Estimator', [BayesianRidge, ARDRegression])
def test_bayesian_ridge_ard_n_iter_deprecated(Estimator):
    """Check the deprecation warning of `n_iter`."""
    depr_msg = "'n_iter' was renamed to 'max_iter' in version 1.3 and will be removed in 1.5"
    X, y = (diabetes.data, diabetes.target)
    model = Estimator(n_iter=5)
    with pytest.warns(FutureWarning, match=depr_msg):
        model.fit(X, y)