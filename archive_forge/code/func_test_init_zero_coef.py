import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize
from sklearn._loss.loss import (
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('base_loss', LOSSES)
@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('n_features', [0, 1, 10])
@pytest.mark.parametrize('dtype', [None, np.float32, np.float64, np.int64])
def test_init_zero_coef(base_loss, fit_intercept, n_features, dtype):
    """Test that init_zero_coef initializes coef correctly."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    rng = np.random.RandomState(42)
    X = rng.normal(size=(5, n_features))
    coef = loss.init_zero_coef(X, dtype=dtype)
    if loss.base_loss.is_multiclass:
        n_classes = loss.base_loss.n_classes
        assert coef.shape == (n_classes, n_features + fit_intercept)
        assert coef.flags['F_CONTIGUOUS']
    else:
        assert coef.shape == (n_features + fit_intercept,)
    if dtype is None:
        assert coef.dtype == X.dtype
    else:
        assert coef.dtype == dtype
    assert np.count_nonzero(coef) == 0