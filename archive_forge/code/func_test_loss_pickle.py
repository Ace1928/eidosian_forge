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
def test_loss_pickle(loss):
    """Test that losses can be pickled."""
    n_samples = 20
    y_true, raw_prediction = random_y_true_raw_prediction(loss=loss, n_samples=n_samples, y_bound=(-100, 100), raw_bound=(-5, 5), seed=42)
    pickled_loss = pickle.dumps(loss)
    unpickled_loss = pickle.loads(pickled_loss)
    assert loss(y_true=y_true, raw_prediction=raw_prediction) == approx(unpickled_loss(y_true=y_true, raw_prediction=raw_prediction))