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
def test_group_shuffle_split():
    for groups_i in test_groups:
        X = y = np.ones(len(groups_i))
        n_splits = 6
        test_size = 1.0 / 3
        slo = GroupShuffleSplit(n_splits, test_size=test_size, random_state=0)
        repr(slo)
        assert slo.get_n_splits(X, y, groups=groups_i) == n_splits
        l_unique = np.unique(groups_i)
        l = np.asarray(groups_i)
        for train, test in slo.split(X, y, groups=groups_i):
            l_train_unique = np.unique(l[train])
            l_test_unique = np.unique(l[test])
            assert not np.any(np.isin(l[train], l_test_unique))
            assert not np.any(np.isin(l[test], l_train_unique))
            assert l[train].size + l[test].size == l.size
            assert_array_equal(np.intersect1d(train, test), [])
            assert abs(len(l_test_unique) - round(test_size * len(l_unique))) <= 1
            assert abs(len(l_train_unique) - round((1.0 - test_size) * len(l_unique))) <= 1