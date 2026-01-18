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
def assert_tree_equal(d, s, message):
    assert s.node_count == d.node_count, '{0}: inequal number of node ({1} != {2})'.format(message, s.node_count, d.node_count)
    assert_array_equal(d.children_right, s.children_right, message + ': inequal children_right')
    assert_array_equal(d.children_left, s.children_left, message + ': inequal children_left')
    external = d.children_right == TREE_LEAF
    internal = np.logical_not(external)
    assert_array_equal(d.feature[internal], s.feature[internal], message + ': inequal features')
    assert_array_equal(d.threshold[internal], s.threshold[internal], message + ': inequal threshold')
    assert_array_equal(d.n_node_samples.sum(), s.n_node_samples.sum(), message + ': inequal sum(n_node_samples)')
    assert_array_equal(d.n_node_samples, s.n_node_samples, message + ': inequal n_node_samples')
    assert_almost_equal(d.impurity, s.impurity, err_msg=message + ': inequal impurity')
    assert_array_almost_equal(d.value[external], s.value[external], err_msg=message + ': inequal value')