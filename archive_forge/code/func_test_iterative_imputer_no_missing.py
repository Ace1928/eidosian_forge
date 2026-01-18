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
def test_iterative_imputer_no_missing():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 100)
    X[:, 0] = np.nan
    m1 = IterativeImputer(max_iter=10, random_state=rng)
    m2 = IterativeImputer(max_iter=10, random_state=rng)
    pred1 = m1.fit(X).transform(X)
    pred2 = m2.fit_transform(X)
    assert_allclose(X[:, 1:], pred1)
    assert_allclose(pred1, pred2)