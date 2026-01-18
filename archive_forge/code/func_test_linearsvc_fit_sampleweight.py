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
def test_linearsvc_fit_sampleweight():
    n_samples = len(X)
    unit_weight = np.ones(n_samples)
    clf = svm.LinearSVC(dual='auto', random_state=0).fit(X, Y)
    clf_unitweight = svm.LinearSVC(dual='auto', random_state=0, tol=1e-12, max_iter=1000).fit(X, Y, sample_weight=unit_weight)
    assert_array_equal(clf_unitweight.predict(T), clf.predict(T))
    assert_allclose(clf.coef_, clf_unitweight.coef_, 1, 0.0001)
    random_state = check_random_state(0)
    random_weight = random_state.randint(0, 10, n_samples)
    lsvc_unflat = svm.LinearSVC(dual='auto', random_state=0, tol=1e-12, max_iter=1000).fit(X, Y, sample_weight=random_weight)
    pred1 = lsvc_unflat.predict(T)
    X_flat = np.repeat(X, random_weight, axis=0)
    y_flat = np.repeat(Y, random_weight, axis=0)
    lsvc_flat = svm.LinearSVC(dual='auto', random_state=0, tol=1e-12, max_iter=1000).fit(X_flat, y_flat)
    pred2 = lsvc_flat.predict(T)
    assert_array_equal(pred1, pred2)
    assert_allclose(lsvc_unflat.coef_, lsvc_flat.coef_, 1, 0.0001)