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
@pytest.mark.parametrize('kfold', [GroupKFold, StratifiedGroupKFold])
def test_group_kfold(kfold):
    rng = np.random.RandomState(0)
    n_groups = 15
    n_samples = 1000
    n_splits = 5
    X = y = np.ones(n_samples)
    tolerance = 0.05 * n_samples
    groups = rng.randint(0, n_groups, n_samples)
    ideal_n_groups_per_fold = n_samples // n_splits
    len(np.unique(groups))
    folds = np.zeros(n_samples)
    lkf = kfold(n_splits=n_splits)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        folds[test] = i
    assert len(folds) == len(groups)
    for i in np.unique(folds):
        assert tolerance >= abs(sum(folds == i) - ideal_n_groups_per_fold)
    for group in np.unique(groups):
        assert len(np.unique(folds[groups == group])) == 1
    groups = np.asarray(groups, dtype=object)
    for train, test in lkf.split(X, y, groups):
        assert len(np.intersect1d(groups[train], groups[test])) == 0
    groups = np.array(['Albert', 'Jean', 'Bertrand', 'Michel', 'Jean', 'Francis', 'Robert', 'Michel', 'Rachel', 'Lois', 'Michelle', 'Bernard', 'Marion', 'Laura', 'Jean', 'Rachel', 'Franck', 'John', 'Gael', 'Anna', 'Alix', 'Robert', 'Marion', 'David', 'Tony', 'Abel', 'Becky', 'Madmood', 'Cary', 'Mary', 'Alexandre', 'David', 'Francis', 'Barack', 'Abdoul', 'Rasha', 'Xi', 'Silvia'])
    n_groups = len(np.unique(groups))
    n_samples = len(groups)
    n_splits = 5
    tolerance = 0.05 * n_samples
    ideal_n_groups_per_fold = n_samples // n_splits
    X = y = np.ones(n_samples)
    folds = np.zeros(n_samples)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        folds[test] = i
    assert len(folds) == len(groups)
    for i in np.unique(folds):
        assert tolerance >= abs(sum(folds == i) - ideal_n_groups_per_fold)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for group in np.unique(groups):
            assert len(np.unique(folds[groups == group])) == 1
    groups = np.asarray(groups, dtype=object)
    for train, test in lkf.split(X, y, groups):
        assert len(np.intersect1d(groups[train], groups[test])) == 0
    cv_iter = list(lkf.split(X, y, groups.tolist()))
    for (train1, test1), (train2, test2) in zip(lkf.split(X, y, groups), cv_iter):
        assert_array_equal(train1, train2)
        assert_array_equal(test1, test2)
    groups = np.array([1, 1, 1, 2, 2])
    X = y = np.ones(len(groups))
    with pytest.raises(ValueError, match='Cannot have number of splits.*greater'):
        next(GroupKFold(n_splits=3).split(X, y, groups))