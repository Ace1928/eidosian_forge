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
@pytest.mark.parametrize('grid_resolution', [3, 100])
def test_grid_from_X_heterogeneous_type(grid_resolution):
    """Check that `_grid_from_X` always sample from categories and does not
    depend from the percentiles.
    """
    pd = pytest.importorskip('pandas')
    percentiles = (0.05, 0.95)
    is_categorical = [True, False]
    X = pd.DataFrame({'cat': ['A', 'B', 'C', 'A', 'B', 'D', 'E', 'A', 'B', 'D'], 'num': [1, 1, 1, 2, 5, 6, 6, 6, 6, 8]})
    nunique = X.nunique()
    grid, axes = _grid_from_X(X, percentiles, is_categorical, grid_resolution=grid_resolution)
    if grid_resolution == 3:
        assert grid.shape == (15, 2)
        assert axes[0].shape[0] == nunique['num']
        assert axes[1].shape[0] == grid_resolution
    else:
        assert grid.shape == (25, 2)
        assert axes[0].shape[0] == nunique['cat']
        assert axes[1].shape[0] == nunique['cat']