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
@pytest.mark.parametrize('loss, y_true, raw_prediction, loss_true, gradient_true, hessian_true', [(HalfSquaredError(), 1.0, 5.0, 8, 4, 1), (AbsoluteError(), 1.0, 5.0, 4.0, 1.0, None), (PinballLoss(quantile=0.5), 1.0, 5.0, 2, 0.5, None), (PinballLoss(quantile=0.25), 1.0, 5.0, 4 * (1 - 0.25), 1 - 0.25, None), (PinballLoss(quantile=0.25), 5.0, 1.0, 4 * 0.25, -0.25, None), (HuberLoss(quantile=0.5, delta=3), 1.0, 5.0, 3 * (4 - 3 / 2), None, None), (HuberLoss(quantile=0.5, delta=3), 1.0, 3.0, 0.5 * 2 ** 2, None, None), (HalfPoissonLoss(), 2.0, np.log(4), 4 - 2 * np.log(4), 4 - 2, 4), (HalfGammaLoss(), 2.0, np.log(4), np.log(4) + 2 / 4, 1 - 2 / 4, 2 / 4), (HalfTweedieLoss(power=3), 2.0, np.log(4), -1 / 4 + 1 / 4 ** 2, None, None), (HalfTweedieLossIdentity(power=1), 2.0, 4.0, 2 - 2 * np.log(2), None, None), (HalfTweedieLossIdentity(power=2), 2.0, 4.0, np.log(2) - 1 / 2, None, None), (HalfTweedieLossIdentity(power=3), 2.0, 4.0, -1 / 4 + 1 / 4 ** 2 + 1 / 2 / 2, None, None), (HalfBinomialLoss(), 0.25, np.log(4), np.log1p(4) - 0.25 * np.log(4), None, None), (HalfBinomialLoss(), 0.0, -1e+20, 0, 0, 0), (HalfBinomialLoss(), 1.0, -1e+20, 1e+20, -1, 0), (HalfBinomialLoss(), 0.0, -1000.0, 0, 0, 0), (HalfBinomialLoss(), 1.0, -1000.0, 1000.0, -1, 0), (HalfBinomialLoss(), 1.0, -37.5, 37.5, -1, 0), (HalfBinomialLoss(), 1.0, -37.0, 37, 1e-16 - 1, 8.533047625744065e-17), (HalfBinomialLoss(), 0.0, -37.0, *[8.533047625744065e-17] * 3), (HalfBinomialLoss(), 1.0, -36.9, 36.9, 1e-16 - 1, 9.430476078526806e-17), (HalfBinomialLoss(), 0.0, -36.9, *[9.430476078526806e-17] * 3), (HalfBinomialLoss(), 0.0, 37.0, 37, 1 - 1e-16, 8.533047625744065e-17), (HalfBinomialLoss(), 1.0, 37.0, *[8.533047625744066e-17] * 3), (HalfBinomialLoss(), 0.0, 37.5, 37.5, 1, 5.175555005801868e-17), (HalfBinomialLoss(), 0.0, 232.8, 232.8, 1, 1.4287342391028437e-101), (HalfBinomialLoss(), 1.0, 1e+20, 0, 0, 0), (HalfBinomialLoss(), 0.0, 1e+20, 1e+20, 1, 0), (HalfBinomialLoss(), 1.0, 232.8, 0, -1.4287342391028437e-101, 1.4287342391028437e-101), (HalfBinomialLoss(), 1.0, 232.9, 0, 0, 0), (HalfBinomialLoss(), 1.0, 1000.0, 0, 0, 0), (HalfBinomialLoss(), 0.0, 1000.0, 1000.0, 1, 0), (HalfMultinomialLoss(n_classes=3), 0.0, [0.2, 0.5, 0.3], logsumexp([0.2, 0.5, 0.3]) - 0.2, None, None), (HalfMultinomialLoss(n_classes=3), 1.0, [0.2, 0.5, 0.3], logsumexp([0.2, 0.5, 0.3]) - 0.5, None, None), (HalfMultinomialLoss(n_classes=3), 2.0, [0.2, 0.5, 0.3], logsumexp([0.2, 0.5, 0.3]) - 0.3, None, None), (HalfMultinomialLoss(n_classes=3), 2.0, [10000.0, 0, 7e-07], logsumexp([10000.0, 0, 7e-07]) - 7e-07, None, None)], ids=loss_instance_name)
def test_loss_on_specific_values(loss, y_true, raw_prediction, loss_true, gradient_true, hessian_true):
    """Test losses, gradients and hessians at specific values."""
    loss1 = loss(y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction]))
    grad1 = loss.gradient(y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction]))
    loss2, grad2 = loss.loss_gradient(y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction]))
    grad3, hess = loss.gradient_hessian(y_true=np.array([y_true]), raw_prediction=np.array([raw_prediction]))
    assert loss1 == approx(loss_true, rel=1e-15, abs=1e-15)
    assert loss2 == approx(loss_true, rel=1e-15, abs=1e-15)
    if gradient_true is not None:
        assert grad1 == approx(gradient_true, rel=1e-15, abs=1e-15)
        assert grad2 == approx(gradient_true, rel=1e-15, abs=1e-15)
        assert grad3 == approx(gradient_true, rel=1e-15, abs=1e-15)
    if hessian_true is not None:
        assert hess == approx(hessian_true, rel=1e-15, abs=1e-15)