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
def test_liblinear_set_coef():
    clf = svm.LinearSVC(dual='auto').fit(iris.data, iris.target)
    values = clf.decision_function(iris.data)
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    values2 = clf.decision_function(iris.data)
    assert_array_almost_equal(values, values2)
    X = [[2, 1], [3, 1], [1, 3], [2, 3]]
    y = [0, 0, 1, 1]
    clf = svm.LinearSVC(dual='auto').fit(X, y)
    values = clf.decision_function(X)
    clf.coef_ = clf.coef_.copy()
    clf.intercept_ = clf.intercept_.copy()
    values2 = clf.decision_function(X)
    assert_array_equal(values, values2)