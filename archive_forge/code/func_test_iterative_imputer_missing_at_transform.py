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
@pytest.mark.parametrize('strategy', ['mean', 'median', 'most_frequent'])
def test_iterative_imputer_missing_at_transform(strategy):
    rng = np.random.RandomState(0)
    n = 100
    d = 10
    X_train = rng.randint(low=0, high=3, size=(n, d))
    X_test = rng.randint(low=0, high=3, size=(n, d))
    X_train[:, 0] = 1
    X_test[0, 0] = 0
    imputer = IterativeImputer(missing_values=0, max_iter=1, initial_strategy=strategy, random_state=rng).fit(X_train)
    initial_imputer = SimpleImputer(missing_values=0, strategy=strategy).fit(X_train)
    assert_allclose(imputer.transform(X_test)[:, 0], initial_imputer.transform(X_test)[:, 0])