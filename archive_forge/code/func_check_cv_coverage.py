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
def check_cv_coverage(cv, X, y, groups, expected_n_splits):
    n_samples = _num_samples(X)
    assert cv.get_n_splits(X, y, groups) == expected_n_splits
    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)
    assert iterations == expected_n_splits
    if n_samples is not None:
        assert collected_test_samples == set(range(n_samples))