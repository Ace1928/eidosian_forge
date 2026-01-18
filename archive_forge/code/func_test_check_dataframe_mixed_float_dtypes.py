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
@pytest.mark.parametrize('dtype', ((np.float64, np.float32), np.float64, None, 'numeric'))
@pytest.mark.parametrize('bool_dtype', ('bool', 'boolean'))
def test_check_dataframe_mixed_float_dtypes(dtype, bool_dtype):
    if bool_dtype == 'boolean':
        pd = importorskip('pandas', minversion='1.0')
    else:
        pd = importorskip('pandas')
    df = pd.DataFrame({'int': [1, 2, 3], 'float': [0, 0.1, 2.1], 'bool': pd.Series([True, False, True], dtype=bool_dtype)}, columns=['int', 'float', 'bool'])
    array = check_array(df, dtype=dtype)
    assert array.dtype == np.float64
    expected_array = np.array([[1.0, 0.0, 1.0], [2.0, 0.1, 0.0], [3.0, 2.1, 1.0]], dtype=float)
    assert_allclose_dense_sparse(array, expected_array)