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
def test_linearsvr():
    diabetes = datasets.load_diabetes()
    lsvr = svm.LinearSVR(C=1000.0, dual='auto').fit(diabetes.data, diabetes.target)
    score1 = lsvr.score(diabetes.data, diabetes.target)
    svr = svm.SVR(kernel='linear', C=1000.0).fit(diabetes.data, diabetes.target)
    score2 = svr.score(diabetes.data, diabetes.target)
    assert_allclose(np.linalg.norm(lsvr.coef_), np.linalg.norm(svr.coef_), 1, 0.0001)
    assert_almost_equal(score1, score2, 2)