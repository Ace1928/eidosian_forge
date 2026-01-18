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
@pytest.mark.parametrize('skip_complete', [True, False])
def test_iterative_imputer_skip_non_missing(skip_complete):
    rng = np.random.RandomState(0)
    X_train = np.array([[5, 2, 2, 1], [10, 1, 2, 7], [3, 1, 1, 1], [8, 4, 2, 2]])
    X_test = np.array([[np.nan, 2, 4, 5], [np.nan, 4, 1, 2], [np.nan, 1, 10, 1]])
    imputer = IterativeImputer(initial_strategy='mean', skip_complete=skip_complete, random_state=rng)
    X_test_est = imputer.fit(X_train).transform(X_test)
    if skip_complete:
        assert_allclose(X_test_est[:, 0], np.mean(X_train[:, 0]))
    else:
        assert_allclose(X_test_est[:, 0], [11, 7, 12], rtol=0.0001)