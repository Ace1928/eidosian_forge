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
@pytest.mark.parametrize('array_constructor', CSR_CONTAINERS + [np.asarray])
def test_imputation_constant_float(array_constructor):
    X = np.array([[np.nan, 1.1, 0, np.nan], [1.2, np.nan, 1.3, np.nan], [0, 0, np.nan, np.nan], [1.4, 1.5, 0, np.nan]])
    X_true = np.array([[-1, 1.1, 0, -1], [1.2, -1, 1.3, -1], [0, 0, -1, -1], [1.4, 1.5, 0, -1]])
    X = array_constructor(X)
    X_true = array_constructor(X_true)
    imputer = SimpleImputer(strategy='constant', fill_value=-1)
    X_trans = imputer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans, X_true)