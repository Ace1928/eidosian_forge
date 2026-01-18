import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('X, missing_values, X_trans_exp', [(np.array([['a', 'b'], ['b', 'a']], dtype=object), 'a', np.array([['b', 'b', True, False], ['b', 'b', False, True]], dtype=object)), (np.array([[np.nan, 1.0], [1.0, np.nan]]), np.nan, np.array([[1.0, 1.0, True, False], [1.0, 1.0, False, True]])), (np.array([[np.nan, 'b'], ['b', np.nan]], dtype=object), np.nan, np.array([['b', 'b', True, False], ['b', 'b', False, True]], dtype=object)), (np.array([[None, 'b'], ['b', None]], dtype=object), None, np.array([['b', 'b', True, False], ['b', 'b', False, True]], dtype=object))])
def test_missing_indicator_with_imputer(X, missing_values, X_trans_exp):
    trans = make_union(SimpleImputer(missing_values=missing_values, strategy='most_frequent'), MissingIndicator(missing_values=missing_values))
    X_trans = trans.fit_transform(X)
    assert_array_equal(X_trans, X_trans_exp)