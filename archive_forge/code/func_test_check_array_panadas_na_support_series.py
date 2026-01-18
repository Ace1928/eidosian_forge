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
def test_check_array_panadas_na_support_series():
    """Check check_array is correct with pd.NA in a series."""
    pd = pytest.importorskip('pandas')
    X_int64 = pd.Series([1, 2, pd.NA], dtype='Int64')
    msg = 'Input contains NaN'
    with pytest.raises(ValueError, match=msg):
        check_array(X_int64, force_all_finite=True, ensure_2d=False)
    X_out = check_array(X_int64, force_all_finite=False, ensure_2d=False)
    assert_allclose(X_out, [1, 2, np.nan])
    assert X_out.dtype == np.float64
    X_out = check_array(X_int64, force_all_finite=False, ensure_2d=False, dtype=np.float32)
    assert_allclose(X_out, [1, 2, np.nan])
    assert X_out.dtype == np.float32