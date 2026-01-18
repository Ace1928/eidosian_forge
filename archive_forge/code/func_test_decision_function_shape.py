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
@pytest.mark.parametrize('SVM', (svm.SVC, svm.NuSVC))
def test_decision_function_shape(SVM):
    clf = SVM(kernel='linear', decision_function_shape='ovr').fit(iris.data, iris.target)
    dec = clf.decision_function(iris.data)
    assert dec.shape == (len(iris.data), 3)
    assert_array_equal(clf.predict(iris.data), np.argmax(dec, axis=1))
    X, y = make_blobs(n_samples=80, centers=5, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = SVM(kernel='linear', decision_function_shape='ovr').fit(X_train, y_train)
    dec = clf.decision_function(X_test)
    assert dec.shape == (len(X_test), 5)
    assert_array_equal(clf.predict(X_test), np.argmax(dec, axis=1))
    clf = SVM(kernel='linear', decision_function_shape='ovo').fit(X_train, y_train)
    dec = clf.decision_function(X_train)
    assert dec.shape == (len(X_train), 10)