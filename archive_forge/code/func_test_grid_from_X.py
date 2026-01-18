import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
def test_grid_from_X():
    percentiles = (0.05, 0.95)
    grid_resolution = 100
    is_categorical = [False, False]
    X = np.asarray([[1, 2], [3, 4]])
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution)
    assert_array_equal(grid, [[1, 2], [1, 4], [3, 2], [3, 4]])
    assert_array_equal(axes, X.T)
    rng = np.random.RandomState(0)
    grid_resolution = 15
    X = rng.normal(size=(20, 2))
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution=grid_resolution)
    assert grid.shape == (grid_resolution * grid_resolution, X.shape[1])
    assert np.asarray(axes).shape == (2, grid_resolution)
    n_unique_values = 12
    X[n_unique_values - 1:, 0] = 12345
    rng.shuffle(X)
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution=grid_resolution)
    assert grid.shape == (n_unique_values * grid_resolution, X.shape[1])
    assert axes[0].shape == (n_unique_values,)
    assert axes[1].shape == (grid_resolution,)