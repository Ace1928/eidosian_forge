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
def test_decision_function_shape_two_class():
    for n_classes in [2, 3]:
        X, y = make_blobs(centers=n_classes, random_state=0)
        for estimator in [svm.SVC, svm.NuSVC]:
            clf = OneVsRestClassifier(estimator(decision_function_shape='ovr')).fit(X, y)
            assert len(clf.predict(X)) == len(y)