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
def test_iterative_imputer_verbose():
    rng = np.random.RandomState(0)
    n = 100
    d = 3
    X = _sparse_random_matrix(n, d, density=0.1, random_state=rng).toarray()
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=1)
    imputer.fit(X)
    imputer.transform(X)
    imputer = IterativeImputer(missing_values=0, max_iter=1, verbose=2)
    imputer.fit(X)
    imputer.transform(X)