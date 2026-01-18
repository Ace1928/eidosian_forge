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
@pytest.mark.parametrize('min_max_1, min_max_2', [([None, None], [-np.inf, np.inf]), ([-10, 10], [[-10] * 4, [10] * 4])], ids=['None-vs-inf', 'Scalar-vs-vector'])
def test_iterative_imputer_min_max_array_like_imputation(min_max_1, min_max_2):
    X_train = np.array([[np.nan, 2, 2, 1], [10, np.nan, np.nan, 7], [3, 1, np.nan, 1], [np.nan, 4, 2, np.nan]])
    X_test = np.array([[np.nan, 2, np.nan, 5], [2, 4, np.nan, np.nan], [np.nan, 1, 10, 1]])
    imputer1 = IterativeImputer(min_value=min_max_1[0], max_value=min_max_1[1], random_state=0)
    imputer2 = IterativeImputer(min_value=min_max_2[0], max_value=min_max_2[1], random_state=0)
    X_test_imputed1 = imputer1.fit(X_train).transform(X_test)
    X_test_imputed2 = imputer2.fit(X_train).transform(X_test)
    assert_allclose(X_test_imputed1[:, 0], X_test_imputed2[:, 0])