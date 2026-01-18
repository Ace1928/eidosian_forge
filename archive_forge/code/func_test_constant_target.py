import re
import sys
import warnings
import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.kernels import (
from sklearn.gaussian_process.tests._mini_sequence_kernel import MiniSeqKernel
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_constant_target(kernel):
    """Check that the std. dev. is affected to 1 when normalizing a constant
    feature.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/18318
    NaN where affected to the target when scaling due to null std. dev. with
    constant target.
    """
    y_constant = np.ones(X.shape[0], dtype=np.float64)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X, y_constant)
    assert gpr._y_train_std == pytest.approx(1.0)
    y_pred, y_cov = gpr.predict(X, return_cov=True)
    assert_allclose(y_pred, y_constant)
    assert_allclose(np.diag(y_cov), 0.0, atol=1e-09)
    n_samples, n_targets = (X.shape[0], 2)
    rng = np.random.RandomState(0)
    y = np.concatenate([rng.normal(size=(n_samples, 1)), np.full(shape=(n_samples, 1), fill_value=2)], axis=1)
    gpr.fit(X, y)
    Y_pred, Y_cov = gpr.predict(X, return_cov=True)
    assert_allclose(Y_pred[:, 1], 2)
    assert_allclose(np.diag(Y_cov[..., 1]), 0.0, atol=1e-09)
    assert Y_pred.shape == (n_samples, n_targets)
    assert Y_cov.shape == (n_samples, n_samples, n_targets)