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
@pytest.mark.parametrize('X_fit, X_trans, params, msg_err', [(np.array([[-1, 1], [1, 2]]), np.array([[-1, 1], [1, -1]]), {'features': 'missing-only', 'sparse': 'auto'}, 'have missing values in transform but have no missing values in fit'), (np.array([['a', 'b'], ['c', 'a']], dtype=str), np.array([['a', 'b'], ['c', 'a']], dtype=str), {}, 'MissingIndicator does not support data with dtype')])
def test_missing_indicator_error(X_fit, X_trans, params, msg_err):
    indicator = MissingIndicator(missing_values=-1)
    indicator.set_params(**params)
    with pytest.raises(ValueError, match=msg_err):
        indicator.fit(X_fit).transform(X_trans)