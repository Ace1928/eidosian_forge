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
@skip_if_array_api_compat_not_configured
@pytest.mark.parametrize('array_namespace', ['numpy.array_api', 'cupy.array_api'])
def test_check_array_array_api_has_non_finite(array_namespace):
    """Checks that Array API arrays checks non-finite correctly."""
    xp = pytest.importorskip(array_namespace)
    X_nan = xp.asarray([[xp.nan, 1, 0], [0, xp.nan, 3]], dtype=xp.float32)
    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match='Input contains NaN.'):
            check_array(X_nan)
    X_inf = xp.asarray([[xp.inf, 1, 0], [0, xp.inf, 3]], dtype=xp.float32)
    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match='infinity or a value too large'):
            check_array(X_inf)