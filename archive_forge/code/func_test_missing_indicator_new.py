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
@pytest.mark.parametrize('arr_type, missing_values, dtype', _generate_missing_indicator_cases())
@pytest.mark.parametrize('param_features, n_features, features_indices', [('missing-only', 3, np.array([0, 1, 2])), ('all', 3, np.array([0, 1, 2]))])
def test_missing_indicator_new(missing_values, arr_type, dtype, param_features, n_features, features_indices):
    X_fit = np.array([[missing_values, missing_values, 1], [4, 2, missing_values]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit_expected = np.array([[1, 1, 0], [0, 0, 1]])
    X_trans_expected = np.array([[1, 1, 0], [0, 0, 0]])
    X_fit = arr_type(X_fit).astype(dtype)
    X_trans = arr_type(X_trans).astype(dtype)
    X_fit_expected = X_fit_expected.astype(dtype)
    X_trans_expected = X_trans_expected.astype(dtype)
    indicator = MissingIndicator(missing_values=missing_values, features=param_features, sparse=False)
    X_fit_mask = indicator.fit_transform(X_fit)
    X_trans_mask = indicator.transform(X_trans)
    assert X_fit_mask.shape[1] == n_features
    assert X_trans_mask.shape[1] == n_features
    assert_array_equal(indicator.features_, features_indices)
    assert_allclose(X_fit_mask, X_fit_expected[:, features_indices])
    assert_allclose(X_trans_mask, X_trans_expected[:, features_indices])
    assert X_fit_mask.dtype == bool
    assert X_trans_mask.dtype == bool
    assert isinstance(X_fit_mask, np.ndarray)
    assert isinstance(X_trans_mask, np.ndarray)
    indicator.set_params(sparse=True)
    X_fit_mask_sparse = indicator.fit_transform(X_fit)
    X_trans_mask_sparse = indicator.transform(X_trans)
    assert X_fit_mask_sparse.dtype == bool
    assert X_trans_mask_sparse.dtype == bool
    assert X_fit_mask_sparse.format == 'csc'
    assert X_trans_mask_sparse.format == 'csc'
    assert_allclose(X_fit_mask_sparse.toarray(), X_fit_mask)
    assert_allclose(X_trans_mask_sparse.toarray(), X_trans_mask)