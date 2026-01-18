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
def test_linearsvc_crammer_singer():
    ovr_clf = svm.LinearSVC(dual='auto', random_state=0).fit(iris.data, iris.target)
    cs_clf = svm.LinearSVC(dual='auto', multi_class='crammer_singer', random_state=0)
    cs_clf.fit(iris.data, iris.target)
    assert (ovr_clf.predict(iris.data) == cs_clf.predict(iris.data)).mean() > 0.9
    assert (ovr_clf.coef_ != cs_clf.coef_).all()
    assert_array_equal(cs_clf.predict(iris.data), np.argmax(cs_clf.decision_function(iris.data), axis=1))
    dec_func = np.dot(iris.data, cs_clf.coef_.T) + cs_clf.intercept_
    assert_array_almost_equal(dec_func, cs_clf.decision_function(iris.data))