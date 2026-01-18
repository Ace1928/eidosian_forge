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
@pytest.mark.parametrize('X', [[[1], [2]], [[1], [np.nan]]])
def test_iterative_imputer_one_feature(X):
    imputer = IterativeImputer().fit(X)
    assert imputer.n_iter_ == 0
    imputer = IterativeImputer()
    imputer.fit([[1], [2]])
    assert imputer.n_iter_ == 0
    imputer.fit([[1], [np.nan]])
    assert imputer.n_iter_ == 0