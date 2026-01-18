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
@pytest.mark.parametrize('imputer_constructor', [SimpleImputer, IterativeImputer])
@pytest.mark.parametrize('imputer_missing_values, missing_value, err_msg', [('NaN', np.nan, 'Input X contains NaN'), ('-1', -1, 'types are expected to be both numerical.')])
def test_inconsistent_dtype_X_missing_values(imputer_constructor, imputer_missing_values, missing_value, err_msg):
    rng = np.random.RandomState(42)
    X = rng.randn(10, 10)
    X[0, 0] = missing_value
    imputer = imputer_constructor(missing_values=imputer_missing_values)
    with pytest.raises(ValueError, match=err_msg):
        imputer.fit_transform(X)