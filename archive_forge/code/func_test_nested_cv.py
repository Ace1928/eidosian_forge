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
def test_nested_cv():
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=15, n_classes=2, random_state=0)
    groups = rng.randint(0, 5, 15)
    cvs = [LeaveOneGroupOut(), StratifiedKFold(n_splits=2), LeaveOneOut(), GroupKFold(n_splits=3), StratifiedKFold(), StratifiedGroupKFold(), StratifiedShuffleSplit(n_splits=3, random_state=0)]
    for inner_cv, outer_cv in combinations_with_replacement(cvs, 2):
        gs = GridSearchCV(DummyClassifier(), param_grid={'strategy': ['stratified', 'most_frequent']}, cv=inner_cv, error_score='raise')
        cross_val_score(gs, X=X, y=y, groups=groups, cv=outer_cv, params={'groups': groups})