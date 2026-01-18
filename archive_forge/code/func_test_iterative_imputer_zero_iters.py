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
def test_iterative_imputer_zero_iters():
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng).toarray()
    missing_flag = X == 0
    X[missing_flag] = np.nan
    imputer = IterativeImputer(max_iter=0)
    X_imputed = imputer.fit_transform(X)
    assert_allclose(X_imputed, imputer.initial_imputer_.transform(X))
    imputer = IterativeImputer(max_iter=5).fit(X)
    assert not np.all(imputer.transform(X) == imputer.initial_imputer_.transform(X))
    imputer.n_iter_ = 0
    assert_allclose(imputer.transform(X), imputer.initial_imputer_.transform(X))