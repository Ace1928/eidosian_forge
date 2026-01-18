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
def test_kfold_valueerrors():
    X1 = np.array([[1, 2], [3, 4], [5, 6]])
    X2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    (ValueError, next, KFold(4).split(X1))
    y = np.array([3, 3, -1, -1, 3])
    skf_3 = StratifiedKFold(3)
    with pytest.warns(Warning, match='The least populated class'):
        next(skf_3.split(X2, y))
    sgkf_3 = StratifiedGroupKFold(3)
    naive_groups = np.arange(len(y))
    with pytest.warns(Warning, match='The least populated class'):
        next(sgkf_3.split(X2, y, naive_groups))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        check_cv_coverage(skf_3, X2, y, groups=None, expected_n_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        check_cv_coverage(sgkf_3, X2, y, groups=naive_groups, expected_n_splits=3)
    y = np.array([3, 3, -1, -1, 2])
    with pytest.raises(ValueError):
        next(skf_3.split(X2, y))
    with pytest.raises(ValueError):
        next(sgkf_3.split(X2, y))
    with pytest.raises(ValueError):
        KFold(0)
    with pytest.raises(ValueError):
        KFold(1)
    error_string = 'k-fold cross-validation requires at least one train/test split'
    with pytest.raises(ValueError, match=error_string):
        StratifiedKFold(0)
    with pytest.raises(ValueError, match=error_string):
        StratifiedKFold(1)
    with pytest.raises(ValueError, match=error_string):
        StratifiedGroupKFold(0)
    with pytest.raises(ValueError, match=error_string):
        StratifiedGroupKFold(1)
    with pytest.raises(ValueError):
        KFold(1.5)
    with pytest.raises(ValueError):
        KFold(2.0)
    with pytest.raises(ValueError):
        StratifiedKFold(1.5)
    with pytest.raises(ValueError):
        StratifiedKFold(2.0)
    with pytest.raises(ValueError):
        StratifiedGroupKFold(1.5)
    with pytest.raises(ValueError):
        StratifiedGroupKFold(2.0)
    with pytest.raises(TypeError):
        KFold(n_splits=4, shuffle=None)