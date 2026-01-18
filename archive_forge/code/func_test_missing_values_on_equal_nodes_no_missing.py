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
@pytest.mark.parametrize('criterion', ['squared_error', 'friedman_mse'])
def test_missing_values_on_equal_nodes_no_missing(criterion):
    """Check missing values goes to correct node during predictions"""
    X = np.array([[0, 1, 2, 3, 8, 9, 11, 12, 15]]).T
    y = np.array([0.1, 0.2, 0.3, 0.2, 1.4, 1.4, 1.5, 1.6, 2.6])
    dtc = DecisionTreeRegressor(random_state=42, max_depth=1, criterion=criterion)
    dtc.fit(X, y)
    y_pred = dtc.predict([[np.nan]])
    assert_allclose(y_pred, [np.mean(y[-5:])])
    X_equal = X[:-1]
    y_equal = y[:-1]
    dtc = DecisionTreeRegressor(random_state=42, max_depth=1, criterion=criterion)
    dtc.fit(X_equal, y_equal)
    y_pred = dtc.predict([[np.nan]])
    assert_allclose(y_pred, [np.mean(y_equal[-4:])])