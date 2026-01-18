import pickle
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx
from scipy.optimize import (
from scipy.special import logsumexp
from sklearn._loss.link import IdentityLink, _inclusive_low_high
from sklearn._loss.loss import (
from sklearn.utils import _IS_WASM, assert_all_finite
from sklearn.utils._testing import create_memmap_backed_data, skip_if_32bit
@pytest.mark.parametrize('loss', LOSS_INSTANCES, ids=loss_instance_name)
def test_graceful_squeezing(loss):
    """Test that reshaped raw_prediction gives same results."""
    y_true, raw_prediction = random_y_true_raw_prediction(loss=loss, n_samples=20, y_bound=(-100, 100), raw_bound=(-10, 10), seed=42)
    if raw_prediction.ndim == 1:
        raw_prediction_2d = raw_prediction[:, None]
        assert_allclose(loss.loss(y_true=y_true, raw_prediction=raw_prediction_2d), loss.loss(y_true=y_true, raw_prediction=raw_prediction))
        assert_allclose(loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction_2d), loss.loss_gradient(y_true=y_true, raw_prediction=raw_prediction))
        assert_allclose(loss.gradient(y_true=y_true, raw_prediction=raw_prediction_2d), loss.gradient(y_true=y_true, raw_prediction=raw_prediction))
        assert_allclose(loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction_2d), loss.gradient_hessian(y_true=y_true, raw_prediction=raw_prediction))