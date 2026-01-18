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
@pytest.mark.parametrize('names', [list(range(2)), range(2), None, [['a', 'b'], ['c', 'd']]], ids=['list-int', 'range', 'default', 'MultiIndex'])
def test_get_feature_names_pandas_with_ints_no_warning(names):
    """Get feature names with pandas dataframes without warning.

    Column names with consistent dtypes will not warn, such as int or MultiIndex.
    """
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        names = _get_feature_names(X)
    assert names is None