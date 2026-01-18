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
@pytest.mark.parametrize('sample_weight', [None, 'range'])
@pytest.mark.parametrize('dtype', (np.float32, np.float64))
@pytest.mark.parametrize('order', ('C', 'F'))
def test_init_gradient_and_hessians(loss, sample_weight, dtype, order):
    """Test that init_gradient_and_hessian works as expected.

    passing sample_weight to a loss correctly influences the constant_hessian
    attribute, and consequently the shape of the hessian array.
    """
    n_samples = 5
    if sample_weight == 'range':
        sample_weight = np.ones(n_samples)
    loss = loss(sample_weight=sample_weight)
    gradient, hessian = loss.init_gradient_and_hessian(n_samples=n_samples, dtype=dtype, order=order)
    if loss.constant_hessian:
        assert gradient.shape == (n_samples,)
        assert hessian.shape == (1,)
    elif loss.is_multiclass:
        assert gradient.shape == (n_samples, loss.n_classes)
        assert hessian.shape == (n_samples, loss.n_classes)
    else:
        assert hessian.shape == (n_samples,)
        assert hessian.shape == (n_samples,)
    assert gradient.dtype == dtype
    assert hessian.dtype == dtype
    if order == 'C':
        assert gradient.flags.c_contiguous
        assert hessian.flags.c_contiguous
    else:
        assert gradient.flags.f_contiguous
        assert hessian.flags.f_contiguous