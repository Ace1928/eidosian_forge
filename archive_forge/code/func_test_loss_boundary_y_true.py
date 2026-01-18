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
@pytest.mark.parametrize('loss, y_true_success, y_true_fail', Y_COMMON_PARAMS + Y_TRUE_PARAMS)
def test_loss_boundary_y_true(loss, y_true_success, y_true_fail):
    """Test boundaries of y_true for loss functions."""
    for y in y_true_success:
        assert loss.in_y_true_range(np.array([y]))
    for y in y_true_fail:
        assert not loss.in_y_true_range(np.array([y]))