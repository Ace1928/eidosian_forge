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
@pytest.mark.parametrize('rank', [3, 5])
def test_iterative_imputer_transform_recovery(rank):
    rng = np.random.RandomState(0)
    n = 70
    d = 70
    A = rng.rand(n, rank)
    B = rng.rand(rank, d)
    X_filled = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X_filled.copy()
    X_missing[nan_mask] = np.nan
    n = n // 2
    X_train = X_missing[:n]
    X_test_filled = X_filled[n:]
    X_test = X_missing[n:]
    imputer = IterativeImputer(max_iter=5, imputation_order='descending', verbose=1, random_state=rng).fit(X_train)
    X_test_est = imputer.transform(X_test)
    assert_allclose(X_test_filled, X_test_est, atol=0.1)