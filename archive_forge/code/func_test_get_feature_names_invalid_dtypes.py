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
@pytest.mark.parametrize('names, dtypes', [(['a', 1], "['int', 'str']"), (['pizza', ['a', 'b']], "['list', 'str']")], ids=['int-str', 'list-str'])
def test_get_feature_names_invalid_dtypes(names, dtypes):
    """Get feature names errors when the feature names have mixed dtypes"""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame([[1, 2], [4, 5], [5, 6]], columns=names)
    msg = re.escape(f'Feature names are only supported if all input features have string names, but your input has {dtypes} as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.')
    with pytest.raises(TypeError, match=msg):
        names = _get_feature_names(X)