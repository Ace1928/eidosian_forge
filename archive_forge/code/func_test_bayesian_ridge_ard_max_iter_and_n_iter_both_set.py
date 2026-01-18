from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
@pytest.mark.parametrize('Estimator', [BayesianRidge, ARDRegression])
def test_bayesian_ridge_ard_max_iter_and_n_iter_both_set(Estimator):
    """Check that a ValueError is raised when both `max_iter` and `n_iter` are set."""
    err_msg = 'Both `n_iter` and `max_iter` attributes were set. Attribute `n_iter` was deprecated in version 1.3 and will be removed in 1.5. To avoid this error, only set the `max_iter` attribute.'
    X, y = (diabetes.data, diabetes.target)
    model = Estimator(n_iter=5, max_iter=5)
    with pytest.raises(ValueError, match=err_msg):
        model.fit(X, y)