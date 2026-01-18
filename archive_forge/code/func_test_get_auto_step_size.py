import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
def test_get_auto_step_size():
    X = np.array([[1, 2, 3], [2, 3, 4], [2, 3, 2]], dtype=np.float64)
    alpha = 1.2
    fit_intercept = False
    max_squared_sum = 4 + 9 + 16
    max_squared_sum_ = row_norms(X, squared=True).max()
    n_samples = X.shape[0]
    assert_almost_equal(max_squared_sum, max_squared_sum_, decimal=4)
    for saga in [True, False]:
        for fit_intercept in (True, False):
            if saga:
                L_sqr = max_squared_sum + alpha + int(fit_intercept)
                L_log = (max_squared_sum + 4.0 * alpha + int(fit_intercept)) / 4.0
                mun_sqr = min(2 * n_samples * alpha, L_sqr)
                mun_log = min(2 * n_samples * alpha, L_log)
                step_size_sqr = 1 / (2 * L_sqr + mun_sqr)
                step_size_log = 1 / (2 * L_log + mun_log)
            else:
                step_size_sqr = 1.0 / (max_squared_sum + alpha + int(fit_intercept))
                step_size_log = 4.0 / (max_squared_sum + 4.0 * alpha + int(fit_intercept))
            step_size_sqr_ = get_auto_step_size(max_squared_sum_, alpha, 'squared', fit_intercept, n_samples=n_samples, is_saga=saga)
            step_size_log_ = get_auto_step_size(max_squared_sum_, alpha, 'log', fit_intercept, n_samples=n_samples, is_saga=saga)
            assert_almost_equal(step_size_sqr, step_size_sqr_, decimal=4)
            assert_almost_equal(step_size_log, step_size_log_, decimal=4)
    msg = 'Unknown loss function for SAG solver, got wrong instead of'
    with pytest.raises(ValueError, match=msg):
        get_auto_step_size(max_squared_sum_, alpha, 'wrong', fit_intercept)