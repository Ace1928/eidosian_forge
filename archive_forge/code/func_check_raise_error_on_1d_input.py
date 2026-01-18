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
@ignore_warnings
def check_raise_error_on_1d_input(name):
    TreeEstimator = ALL_TREES[name]
    X = iris.data[:, 0].ravel()
    X_2d = iris.data[:, 0].reshape((-1, 1))
    y = iris.target
    with pytest.raises(ValueError):
        TreeEstimator(random_state=0).fit(X, y)
    est = TreeEstimator(random_state=0)
    est.fit(X_2d, y)
    with pytest.raises(ValueError):
        est.predict([X])