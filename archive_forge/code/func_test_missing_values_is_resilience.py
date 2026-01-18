import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('make_data,Tree', [(datasets.make_friedman1, DecisionTreeRegressor), (make_friedman1_classification, DecisionTreeClassifier)])
@pytest.mark.parametrize('sample_weight_train', [None, 'ones'])
def test_missing_values_is_resilience(make_data, Tree, sample_weight_train, global_random_seed):
    """Check that trees can deal with missing values have decent performance."""
    n_samples, n_features = (5000, 10)
    X, y = make_data(n_samples=n_samples, n_features=n_features, random_state=global_random_seed)
    X_missing = X.copy()
    rng = np.random.RandomState(global_random_seed)
    X_missing[rng.choice([False, True], size=X.shape, p=[0.9, 0.1])] = np.nan
    X_missing_train, X_missing_test, y_train, y_test = train_test_split(X_missing, y, random_state=global_random_seed)
    if sample_weight_train == 'ones':
        sample_weight = np.ones(X_missing_train.shape[0])
    else:
        sample_weight = None
    native_tree = Tree(max_depth=10, random_state=global_random_seed)
    native_tree.fit(X_missing_train, y_train, sample_weight=sample_weight)
    score_native_tree = native_tree.score(X_missing_test, y_test)
    tree_with_imputer = make_pipeline(SimpleImputer(), Tree(max_depth=10, random_state=global_random_seed))
    tree_with_imputer.fit(X_missing_train, y_train)
    score_tree_with_imputer = tree_with_imputer.score(X_missing_test, y_test)
    assert score_native_tree > score_tree_with_imputer, f'score_native_tree={score_native_tree!r} should be strictly greater than {score_tree_with_imputer}'