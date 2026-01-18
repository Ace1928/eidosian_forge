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
@pytest.mark.parametrize('y_true', (np.array([0.0, 0, 0]), np.array([1.0, 1, 1])))
@pytest.mark.parametrize('y_pred', (np.array([-5.0, -5, -5]), np.array([3.0, 3, 3])))
def test_binomial_vs_alternative_formulation(y_true, y_pred, global_dtype):
    """Test that both formulations of the binomial deviance agree.

    Often, the binomial deviance or log loss is written in terms of a variable
    z in {-1, +1}, but we use y in {0, 1}, hence z = 2 * y - 1.
    ESL II Eq. (10.18):

        -loglike(z, f) = log(1 + exp(-2 * z * f))

    Note:
        - ESL 2*f = raw_prediction, hence the factor 2 of ESL disappears.
        - Deviance = -2*loglike + .., but HalfBinomialLoss is half of the
          deviance, hence the factor of 2 cancels in the comparison.
    """

    def alt_loss(y, raw_pred):
        z = 2 * y - 1
        return np.mean(np.log(1 + np.exp(-z * raw_pred)))

    def alt_gradient(y, raw_pred):
        z = 2 * y - 1
        return -z / (1 + np.exp(z * raw_pred))
    bin_loss = HalfBinomialLoss()
    y_true = y_true.astype(global_dtype)
    y_pred = y_pred.astype(global_dtype)
    datum = (y_true, y_pred)
    assert bin_loss(*datum) == approx(alt_loss(*datum))
    assert_allclose(bin_loss.gradient(*datum), alt_gradient(*datum))