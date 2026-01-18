import re
import numpy as np
import pytest
from numpy.testing import (
from sklearn import base, datasets, linear_model, metrics, svm
from sklearn.datasets import make_blobs, make_classification
from sklearn.exceptions import (
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import (  # type: ignore
from sklearn.svm._classes import _validate_dual_parameter
from sklearn.utils import check_random_state, shuffle
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_linearsvr_fit_sampleweight():
    diabetes = datasets.load_diabetes()
    n_samples = len(diabetes.target)
    unit_weight = np.ones(n_samples)
    lsvr = svm.LinearSVR(dual='auto', C=1000.0, tol=1e-12, max_iter=10000).fit(diabetes.data, diabetes.target, sample_weight=unit_weight)
    score1 = lsvr.score(diabetes.data, diabetes.target)
    lsvr_no_weight = svm.LinearSVR(dual='auto', C=1000.0, tol=1e-12, max_iter=10000).fit(diabetes.data, diabetes.target)
    score2 = lsvr_no_weight.score(diabetes.data, diabetes.target)
    assert_allclose(np.linalg.norm(lsvr.coef_), np.linalg.norm(lsvr_no_weight.coef_), 1, 0.0001)
    assert_almost_equal(score1, score2, 2)
    random_state = check_random_state(0)
    random_weight = random_state.randint(0, 10, n_samples)
    lsvr_unflat = svm.LinearSVR(dual='auto', C=1000.0, tol=1e-12, max_iter=10000).fit(diabetes.data, diabetes.target, sample_weight=random_weight)
    score3 = lsvr_unflat.score(diabetes.data, diabetes.target, sample_weight=random_weight)
    X_flat = np.repeat(diabetes.data, random_weight, axis=0)
    y_flat = np.repeat(diabetes.target, random_weight, axis=0)
    lsvr_flat = svm.LinearSVR(dual='auto', C=1000.0, tol=1e-12, max_iter=10000).fit(X_flat, y_flat)
    score4 = lsvr_flat.score(X_flat, y_flat)
    assert_almost_equal(score3, score4, 2)