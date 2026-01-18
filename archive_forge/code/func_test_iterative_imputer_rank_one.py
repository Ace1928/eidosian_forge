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
def test_iterative_imputer_rank_one():
    rng = np.random.RandomState(0)
    d = 50
    A = rng.rand(d, 1)
    B = rng.rand(1, d)
    X = np.dot(A, B)
    nan_mask = rng.rand(d, d) < 0.5
    X_missing = X.copy()
    X_missing[nan_mask] = np.nan
    imputer = IterativeImputer(max_iter=5, verbose=1, random_state=rng)
    X_filled = imputer.fit_transform(X_missing)
    assert_allclose(X_filled, X, atol=0.02)