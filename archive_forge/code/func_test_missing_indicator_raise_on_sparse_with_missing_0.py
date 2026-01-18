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
@pytest.mark.parametrize('arr_type', CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS)
def test_missing_indicator_raise_on_sparse_with_missing_0(arr_type):
    missing_values = 0
    X_fit = np.array([[missing_values, missing_values, 1], [4, missing_values, 2]])
    X_trans = np.array([[missing_values, missing_values, 1], [4, 12, 10]])
    X_fit_sparse = arr_type(X_fit)
    X_trans_sparse = arr_type(X_trans)
    indicator = MissingIndicator(missing_values=missing_values)
    with pytest.raises(ValueError, match='Sparse input with missing_values=0'):
        indicator.fit_transform(X_fit_sparse)
    indicator.fit_transform(X_fit)
    with pytest.raises(ValueError, match='Sparse input with missing_values=0'):
        indicator.transform(X_trans_sparse)