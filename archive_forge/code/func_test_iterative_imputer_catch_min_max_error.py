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
@pytest.mark.parametrize('min_value, max_value, err_msg', [(100, 0, 'min_value >= max_value.'), (np.inf, -np.inf, 'min_value >= max_value.'), ([-5, 5], [100, 200, 0], "_value' should be of shape")])
def test_iterative_imputer_catch_min_max_error(min_value, max_value, err_msg):
    X = np.random.random((10, 3))
    imputer = IterativeImputer(min_value=min_value, max_value=max_value)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit(X)