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
@pytest.mark.parametrize('loss, y_pred_success, y_pred_fail', Y_COMMON_PARAMS + Y_PRED_PARAMS)
def test_loss_boundary_y_pred(loss, y_pred_success, y_pred_fail):
    """Test boundaries of y_pred for loss functions."""
    for y in y_pred_success:
        assert loss.in_y_pred_range(np.array([y]))
    for y in y_pred_fail:
        assert not loss.in_y_pred_range(np.array([y]))