import math
import re
import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn._loss.loss import HalfMultinomialLoss
from sklearn.base import clone
from sklearn.datasets import load_iris, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model._base import make_dataset
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.linear_model._sag import get_auto_step_size
from sklearn.linear_model._sag_fast import _multinomial_grad_loss_all_samples
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.utils._testing import (
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import CSR_CONTAINERS
def test_multinomial_loss_ground_truth():
    n_classes = 3
    X = np.array([[1.1, 2.2], [2.2, -4.4], [3.3, -2.2], [1.1, 1.1]])
    y = np.array([0, 1, 2, 0], dtype=np.float64)
    lbin = LabelBinarizer()
    Y_bin = lbin.fit_transform(y)
    weights = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, -1.3]])
    intercept = np.array([1.0, 0, -0.2])
    sample_weights = np.array([0.8, 1, 1, 0.8])
    prediction = np.dot(X, weights) + intercept
    logsumexp_prediction = logsumexp(prediction, axis=1)
    p = prediction - logsumexp_prediction[:, np.newaxis]
    loss_1 = -(sample_weights[:, np.newaxis] * p * Y_bin).sum()
    diff = sample_weights[:, np.newaxis] * (np.exp(p) - Y_bin)
    grad_1 = np.dot(X.T, diff)
    loss = LinearModelLoss(base_loss=HalfMultinomialLoss(n_classes=n_classes), fit_intercept=True)
    weights_intercept = np.vstack((weights, intercept)).T
    loss_2, grad_2 = loss.loss_gradient(weights_intercept, X, y, l2_reg_strength=0.0, sample_weight=sample_weights)
    grad_2 = grad_2[:, :-1].T
    loss_2 *= np.sum(sample_weights)
    grad_2 *= np.sum(sample_weights)
    assert_almost_equal(loss_1, loss_2)
    assert_array_almost_equal(grad_1, grad_2)
    loss_gt = 11.68036035432596
    grad_gt = np.array([[-0.557487, -1.619151, +2.176638], [-0.903942, +5.258745, -4.354803]])
    assert_almost_equal(loss_1, loss_gt)
    assert_array_almost_equal(grad_1, grad_gt)