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
@pytest.mark.filterwarnings('ignore:The max_iter was reached')
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_binary_classifier_class_weight(csr_container):
    """tests binary classifier with classweights for each class"""
    alpha = 0.1
    n_samples = 50
    n_iter = 20
    tol = 1e-05
    fit_intercept = True
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=10, cluster_std=0.1)
    step_size = get_step_size(X, alpha, fit_intercept, classification=True)
    classes = np.unique(y)
    y_tmp = np.ones(n_samples)
    y_tmp[y != classes[1]] = -1
    y = y_tmp
    class_weight = {1: 0.45, -1: 0.55}
    clf1 = LogisticRegression(solver='sag', C=1.0 / alpha / n_samples, max_iter=n_iter, tol=tol, random_state=77, fit_intercept=fit_intercept, multi_class='ovr', class_weight=class_weight)
    clf2 = clone(clf1)
    clf1.fit(X, y)
    clf2.fit(csr_container(X), y)
    le = LabelEncoder()
    class_weight_ = compute_class_weight(class_weight, classes=np.unique(y), y=y)
    sample_weight = class_weight_[le.fit_transform(y)]
    spweights, spintercept = sag_sparse(X, y, step_size, alpha, n_iter=n_iter, dloss=log_dloss, sample_weight=sample_weight, fit_intercept=fit_intercept)
    spweights2, spintercept2 = sag_sparse(X, y, step_size, alpha, n_iter=n_iter, dloss=log_dloss, sparse=True, sample_weight=sample_weight, fit_intercept=fit_intercept)
    assert_array_almost_equal(clf1.coef_.ravel(), spweights.ravel(), decimal=2)
    assert_almost_equal(clf1.intercept_, spintercept, decimal=1)
    assert_array_almost_equal(clf2.coef_.ravel(), spweights2.ravel(), decimal=2)
    assert_almost_equal(clf2.intercept_, spintercept2, decimal=1)