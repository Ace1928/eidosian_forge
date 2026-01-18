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
def test_huge_allocations():
    n_bits = 8 * struct.calcsize('P')
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, 10)
    huge = 2 ** (n_bits + 1)
    clf = DecisionTreeClassifier(splitter='best', max_leaf_nodes=huge)
    with pytest.raises(Exception):
        clf.fit(X, y)
    huge = 2 ** (n_bits - 1) - 1
    clf = DecisionTreeClassifier(splitter='best', max_leaf_nodes=huge)
    with pytest.raises(MemoryError):
        clf.fit(X, y)