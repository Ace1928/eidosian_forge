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
@pytest.mark.parametrize('grid_resolution, percentiles, err_msg', [(2, (0, 0.0001), 'percentiles are too close'), (100, (1, 2, 3, 4), "'percentiles' must be a sequence of 2 elements"), (100, 12345, "'percentiles' must be a sequence of 2 elements"), (100, (-1, 0.95), "'percentiles' values must be in \\[0, 1\\]"), (100, (0.05, 2), "'percentiles' values must be in \\[0, 1\\]"), (100, (0.9, 0.1), 'percentiles\\[0\\] must be strictly less than'), (1, (0.05, 0.95), "'grid_resolution' must be strictly greater than 1")])
def test_grid_from_X_error(grid_resolution, percentiles, err_msg):
    X = np.asarray([[1, 2], [3, 4]])
    is_categorical = [False]
    with pytest.raises(ValueError, match=err_msg):
        _grid_from_X(X, percentiles, is_categorical, grid_resolution)