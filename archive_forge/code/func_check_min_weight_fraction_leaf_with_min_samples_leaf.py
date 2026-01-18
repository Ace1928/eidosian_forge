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
def check_min_weight_fraction_leaf_with_min_samples_leaf(name, datasets, sparse_container=None):
    """Test the interaction between min_weight_fraction_leaf and
    min_samples_leaf when sample_weights is not provided in fit."""
    X = DATASETS[datasets]['X'].astype(np.float32)
    if sparse_container is not None:
        X = sparse_container(X)
    y = DATASETS[datasets]['y']
    total_weight = X.shape[0]
    TreeEstimator = ALL_TREES[name]
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 3)):
        est = TreeEstimator(min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=5, random_state=0)
        est.fit(X, y)
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)
        node_weights = np.bincount(out)
        leaf_weights = node_weights[node_weights != 0]
        assert np.min(leaf_weights) >= max(total_weight * est.min_weight_fraction_leaf, 5), 'Failed with {0} min_weight_fraction_leaf={1}, min_samples_leaf={2}'.format(name, est.min_weight_fraction_leaf, est.min_samples_leaf)
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 3)):
        est = TreeEstimator(min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=0.1, random_state=0)
        est.fit(X, y)
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)
        node_weights = np.bincount(out)
        leaf_weights = node_weights[node_weights != 0]
        assert np.min(leaf_weights) >= max(total_weight * est.min_weight_fraction_leaf, total_weight * est.min_samples_leaf), 'Failed with {0} min_weight_fraction_leaf={1}, min_samples_leaf={2}'.format(name, est.min_weight_fraction_leaf, est.min_samples_leaf)