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
def test_iterative_imputer_transform_stochasticity():
    rng1 = np.random.RandomState(0)
    rng2 = np.random.RandomState(1)
    n = 100
    d = 10
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng1).toarray()
    imputer = IterativeImputer(missing_values=0, max_iter=1, sample_posterior=True, random_state=rng1)
    imputer.fit(X)
    X_fitted_1 = imputer.transform(X)
    X_fitted_2 = imputer.transform(X)
    assert np.mean(X_fitted_1) != pytest.approx(np.mean(X_fitted_2))
    imputer1 = IterativeImputer(missing_values=0, max_iter=1, sample_posterior=False, n_nearest_features=None, imputation_order='ascending', random_state=rng1)
    imputer2 = IterativeImputer(missing_values=0, max_iter=1, sample_posterior=False, n_nearest_features=None, imputation_order='ascending', random_state=rng2)
    imputer1.fit(X)
    imputer2.fit(X)
    X_fitted_1a = imputer1.transform(X)
    X_fitted_1b = imputer1.transform(X)
    X_fitted_2 = imputer2.transform(X)
    assert_allclose(X_fitted_1a, X_fitted_1b)
    assert_allclose(X_fitted_1a, X_fitted_2)