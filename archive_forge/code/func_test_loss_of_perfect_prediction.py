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
@pytest.mark.parametrize('sample_weight', [None, 'range'])
def test_loss_of_perfect_prediction(loss, sample_weight):
    """Test value of perfect predictions.

    Loss of y_pred = y_true plus constant_to_optimal_zero should sums up to
    zero.
    """
    if not loss.is_multiclass:
        raw_prediction = np.array([-10, -0.1, 0, 0.1, 3, 10])
        if isinstance(loss.link, IdentityLink):
            eps = 1e-10
            low = loss.interval_y_pred.low
            if not loss.interval_y_pred.low_inclusive:
                low = low + eps
            high = loss.interval_y_pred.high
            if not loss.interval_y_pred.high_inclusive:
                high = high - eps
            raw_prediction = np.clip(raw_prediction, low, high)
        y_true = loss.link.inverse(raw_prediction)
    else:
        y_true = np.arange(loss.n_classes).astype(float)
        raw_prediction = np.full(shape=(loss.n_classes, loss.n_classes), fill_value=-np.exp(10), dtype=float)
        raw_prediction.flat[::loss.n_classes + 1] = np.exp(10)
    if sample_weight == 'range':
        sample_weight = np.linspace(1, y_true.shape[0], num=y_true.shape[0])
    loss_value = loss.loss(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight)
    constant_term = loss.constant_to_optimal_zero(y_true=y_true, sample_weight=sample_weight)
    assert_allclose(loss_value, -constant_term, atol=1e-14, rtol=1e-15)