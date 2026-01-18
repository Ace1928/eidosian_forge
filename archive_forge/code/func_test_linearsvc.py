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
def test_linearsvc():
    clf = svm.LinearSVC(dual='auto', random_state=0).fit(X, Y)
    assert clf.fit_intercept
    assert_array_equal(clf.predict(T), true_result)
    assert_array_almost_equal(clf.intercept_, [0], decimal=3)
    clf = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, random_state=0).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)
    clf = svm.LinearSVC(penalty='l2', dual=True, random_state=0).fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)
    clf = svm.LinearSVC(penalty='l2', loss='hinge', dual=True, random_state=0)
    clf.fit(X, Y)
    assert_array_equal(clf.predict(T), true_result)
    dec = clf.decision_function(T)
    res = (dec > 0).astype(int) + 1
    assert_array_equal(res, true_result)