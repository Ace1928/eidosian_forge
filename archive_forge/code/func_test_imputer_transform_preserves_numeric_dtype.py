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
@pytest.mark.parametrize('dtype_test', [np.float32, np.float64])
def test_imputer_transform_preserves_numeric_dtype(dtype_test):
    """Check transform preserves numeric dtype independent of fit dtype."""
    X = np.asarray([[1.2, 3.4, np.nan], [np.nan, 1.2, 1.3], [4.2, 2, 1]], dtype=np.float64)
    imp = SimpleImputer().fit(X)
    X_test = np.asarray([[np.nan, np.nan, np.nan]], dtype=dtype_test)
    X_trans = imp.transform(X_test)
    assert X_trans.dtype == dtype_test