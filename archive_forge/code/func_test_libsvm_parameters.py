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
def test_libsvm_parameters():
    clf = svm.SVC(kernel='linear').fit(X, Y)
    assert_array_equal(clf.dual_coef_, [[-0.25, 0.25]])
    assert_array_equal(clf.support_, [1, 3])
    assert_array_equal(clf.support_vectors_, (X[1], X[3]))
    assert_array_equal(clf.intercept_, [0.0])
    assert_array_equal(clf.predict(X), Y)