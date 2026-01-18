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
def test_imputation_constant_integer():
    X = np.array([[-1, 2, 3, -1], [4, -1, 5, -1], [6, 7, -1, -1], [8, 9, 0, -1]])
    X_true = np.array([[0, 2, 3, 0], [4, 0, 5, 0], [6, 7, 0, 0], [8, 9, 0, 0]])
    imputer = SimpleImputer(missing_values=-1, strategy='constant', fill_value=0)
    X_trans = imputer.fit_transform(X)
    assert_array_equal(X_trans, X_true)