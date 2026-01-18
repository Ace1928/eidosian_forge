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
def test_leave_group_out_changing_groups():
    groups = np.array([0, 1, 2, 1, 1, 2, 0, 0])
    X = np.ones(len(groups))
    groups_changing = np.array(groups, copy=True)
    lolo = LeaveOneGroupOut().split(X, groups=groups)
    lolo_changing = LeaveOneGroupOut().split(X, groups=groups)
    lplo = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    lplo_changing = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    groups_changing[:] = 0
    for llo, llo_changing in [(lolo, lolo_changing), (lplo, lplo_changing)]:
        for (train, test), (train_chan, test_chan) in zip(llo, llo_changing):
            assert_array_equal(train, train_chan)
            assert_array_equal(test, test_chan)
    assert 3 == LeavePGroupsOut(n_groups=2).get_n_splits(X, y=X, groups=groups)
    assert 3 == LeaveOneGroupOut().get_n_splits(X, y=X, groups=groups)