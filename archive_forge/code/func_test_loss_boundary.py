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
def test_loss_boundary(loss):
    """Test interval ranges of y_true and y_pred in losses."""
    if loss.is_multiclass:
        y_true = np.linspace(0, 9, num=10)
    else:
        low, high = _inclusive_low_high(loss.interval_y_true)
        y_true = np.linspace(low, high, num=10)
    if loss.interval_y_true.low_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.low]
    if loss.interval_y_true.high_inclusive:
        y_true = np.r_[y_true, loss.interval_y_true.high]
    assert loss.in_y_true_range(y_true)
    n = y_true.shape[0]
    low, high = _inclusive_low_high(loss.interval_y_pred)
    if loss.is_multiclass:
        y_pred = np.empty((n, 3))
        y_pred[:, 0] = np.linspace(low, high, num=n)
        y_pred[:, 1] = 0.5 * (1 - y_pred[:, 0])
        y_pred[:, 2] = 0.5 * (1 - y_pred[:, 0])
    else:
        y_pred = np.linspace(low, high, num=n)
    assert loss.in_y_pred_range(y_pred)
    raw_prediction = loss.link.link(y_pred)
    loss.loss(y_true=y_true, raw_prediction=raw_prediction)