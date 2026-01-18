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
def test_iterative_imputer_early_stopping():
    rng = np.random.RandomState(0)
    n = 50
    d = 5
    A = rng.rand(n, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(n, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan
    imputer = IterativeImputer(max_iter=100, tol=0.01, sample_posterior=False, verbose=1, random_state=rng)
    X_filled_100 = imputer.fit_transform(X_missing)
    assert len(imputer.imputation_sequence_) == d * imputer.n_iter_
    imputer = IterativeImputer(max_iter=imputer.n_iter_, sample_posterior=False, verbose=1, random_state=rng)
    X_filled_early = imputer.fit_transform(X_missing)
    assert_allclose(X_filled_100, X_filled_early, atol=1e-07)
    imputer = IterativeImputer(max_iter=100, tol=0, sample_posterior=False, verbose=1, random_state=rng)
    imputer.fit(X_missing)
    assert imputer.n_iter_ == imputer.max_iter