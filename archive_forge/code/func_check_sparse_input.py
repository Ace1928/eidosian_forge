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
def check_sparse_input(tree, dataset, max_depth=None):
    TreeEstimator = ALL_TREES[tree]
    X = DATASETS[dataset]['X']
    y = DATASETS[dataset]['y']
    if dataset in ['digits', 'diabetes']:
        n_samples = X.shape[0] // 5
        X = X[:n_samples]
        y = y[:n_samples]
    for sparse_container in COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS:
        X_sparse = sparse_container(X)
        d = TreeEstimator(random_state=0, max_depth=max_depth).fit(X, y)
        s = TreeEstimator(random_state=0, max_depth=max_depth).fit(X_sparse, y)
        assert_tree_equal(d.tree_, s.tree_, '{0} with dense and sparse format gave different trees'.format(tree))
        y_pred = d.predict(X)
        if tree in CLF_TREES:
            y_proba = d.predict_proba(X)
            y_log_proba = d.predict_log_proba(X)
        for sparse_container_test in COO_CONTAINERS + CSR_CONTAINERS + CSC_CONTAINERS:
            X_sparse_test = sparse_container_test(X_sparse, dtype=np.float32)
            assert_array_almost_equal(s.predict(X_sparse_test), y_pred)
            if tree in CLF_TREES:
                assert_array_almost_equal(s.predict_proba(X_sparse_test), y_proba)
                assert_array_almost_equal(s.predict_log_proba(X_sparse_test), y_log_proba)