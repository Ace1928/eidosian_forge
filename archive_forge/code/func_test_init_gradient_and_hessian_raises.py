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
@pytest.mark.parametrize('loss', ALL_LOSSES)
@pytest.mark.parametrize('params, err_msg', [({'dtype': np.int64}, f"Valid options for 'dtype' are .* Got dtype={np.int64} instead.")])
def test_init_gradient_and_hessian_raises(loss, params, err_msg):
    """Test that init_gradient_and_hessian raises errors for invalid input."""
    loss = loss()
    with pytest.raises((ValueError, TypeError), match=err_msg):
        gradient, hessian = loss.init_gradient_and_hessian(n_samples=5, **params)