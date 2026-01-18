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
@pytest.mark.parametrize('constructor_name, minversion', [('pyarrow', '12.0.0'), ('dataframe', '1.5.0'), ('polars', '0.18.2')])
def test_get_feature_names_dataframe_protocol(constructor_name, minversion):
    """Uses the dataframe exchange protocol to get feature names."""
    data = [[1, 4, 2], [3, 3, 6]]
    columns = ['col_0', 'col_1', 'col_2']
    df = _convert_container(data, constructor_name, columns_name=columns, minversion=minversion)
    feature_names = _get_feature_names(df)
    assert_array_equal(feature_names, columns)