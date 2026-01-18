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
def test_leave_one_p_group_out_error_on_fewer_number_of_groups():
    X = y = groups = np.ones(0)
    msg = re.escape('Found array with 0 sample(s)')
    with pytest.raises(ValueError, match=msg):
        next(LeaveOneGroupOut().split(X, y, groups))
    X = y = groups = np.ones(1)
    msg = re.escape(f'The groups parameter contains fewer than 2 unique groups ({groups}). LeaveOneGroupOut expects at least 2.')
    with pytest.raises(ValueError, match=msg):
        next(LeaveOneGroupOut().split(X, y, groups))
    X = y = groups = np.ones(1)
    msg = re.escape(f'The groups parameter contains fewer than (or equal to) n_groups (3) numbers of unique groups ({groups}). LeavePGroupsOut expects that at least n_groups + 1 (4) unique groups be present')
    with pytest.raises(ValueError, match=msg):
        next(LeavePGroupsOut(n_groups=3).split(X, y, groups))
    X = y = groups = np.arange(3)
    msg = re.escape(f'The groups parameter contains fewer than (or equal to) n_groups (3) numbers of unique groups ({groups}). LeavePGroupsOut expects that at least n_groups + 1 (4) unique groups be present')
    with pytest.raises(ValueError, match=msg):
        next(LeavePGroupsOut(n_groups=3).split(X, y, groups))