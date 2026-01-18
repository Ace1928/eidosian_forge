import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
def test_check_array_pandas_dtype_casting():
    pd = pytest.importorskip('pandas')
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    X_df = pd.DataFrame(X)
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32
    X_df = X_df.astype({0: np.float16})
    assert_array_equal(X_df.dtypes, (np.float16, np.float32, np.float32))
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32
    X_df = X_df.astype({0: np.int16})
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32
    X_df = X_df.astype({2: np.float16})
    assert check_array(X_df).dtype == np.float32
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float32
    X_df = X_df.astype(np.int16)
    assert check_array(X_df).dtype == np.int16
    assert check_array(X_df, dtype=FLOAT_DTYPES).dtype == np.float64
    cat_df = pd.DataFrame({'cat_col': pd.Categorical([1, 2, 3])})
    assert check_array(cat_df).dtype == np.int64
    assert check_array(cat_df, dtype=FLOAT_DTYPES).dtype == np.float64