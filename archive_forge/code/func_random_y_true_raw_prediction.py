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
def random_y_true_raw_prediction(loss, n_samples, y_bound=(-100, 100), raw_bound=(-5, 5), seed=42):
    """Random generate y_true and raw_prediction in valid range."""
    rng = np.random.RandomState(seed)
    if loss.is_multiclass:
        raw_prediction = np.empty((n_samples, loss.n_classes))
        raw_prediction.flat[:] = rng.uniform(low=raw_bound[0], high=raw_bound[1], size=n_samples * loss.n_classes)
        y_true = np.arange(n_samples).astype(float) % loss.n_classes
    else:
        if isinstance(loss.link, IdentityLink):
            low, high = _inclusive_low_high(loss.interval_y_pred)
            low = np.amax([low, raw_bound[0]])
            high = np.amin([high, raw_bound[1]])
            raw_bound = (low, high)
        raw_prediction = rng.uniform(low=raw_bound[0], high=raw_bound[1], size=n_samples)
        low, high = _inclusive_low_high(loss.interval_y_true)
        low = max(low, y_bound[0])
        high = min(high, y_bound[1])
        y_true = rng.uniform(low, high, size=n_samples)
        if loss.interval_y_true.low == 0 and loss.interval_y_true.low_inclusive:
            y_true[::n_samples // 3] = 0
        if loss.interval_y_true.high == 1 and loss.interval_y_true.high_inclusive:
            y_true[1::n_samples // 3] = 1
    return (y_true, raw_prediction)