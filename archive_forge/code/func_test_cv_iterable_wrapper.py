import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_cv_iterable_wrapper():
    kf_iter = KFold().split(X, y)
    kf_iter_wrapped = check_cv(kf_iter)
    np.testing.assert_equal(list(kf_iter_wrapped.split(X, y)), list(kf_iter_wrapped.split(X, y)))
    kf_randomized_iter = KFold(shuffle=True, random_state=0).split(X, y)
    kf_randomized_iter_wrapped = check_cv(kf_randomized_iter)
    np.testing.assert_equal(list(kf_randomized_iter_wrapped.split(X, y)), list(kf_randomized_iter_wrapped.split(X, y)))
    try:
        splits_are_equal = True
        np.testing.assert_equal(list(kf_iter_wrapped.split(X, y)), list(kf_randomized_iter_wrapped.split(X, y)))
    except AssertionError:
        splits_are_equal = False
    assert not splits_are_equal, 'If the splits are randomized, successive calls to split should yield different results'