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
def test_leave_group_out_order_dependence():
    groups = np.array([2, 2, 0, 0, 1, 1])
    X = np.ones(len(groups))
    splits = iter(LeaveOneGroupOut().split(X, groups=groups))
    expected_indices = [([0, 1, 4, 5], [2, 3]), ([0, 1, 2, 3], [4, 5]), ([2, 3, 4, 5], [0, 1])]
    for expected_train, expected_test in expected_indices:
        train, test = next(splits)
        assert_array_equal(train, expected_train)
        assert_array_equal(test, expected_test)