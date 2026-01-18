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
@pytest.mark.parametrize('strategy', ['mean', 'median'])
@pytest.mark.parametrize('dtype', [None, object, str])
def test_imputation_mean_median_error_invalid_type(strategy, dtype):
    X = np.array([['a', 'b', 3], [4, 'e', 6], ['g', 'h', 9]], dtype=dtype)
    msg = 'non-numeric data:\ncould not convert string to float:'
    with pytest.raises(ValueError, match=msg):
        imputer = SimpleImputer(strategy=strategy)
        imputer.fit_transform(X)