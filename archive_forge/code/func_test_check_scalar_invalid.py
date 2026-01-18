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
@pytest.mark.parametrize('x, target_name, target_type, min_val, max_val, include_boundaries, err_msg', [(1, 'test_name1', float, 2, 4, 'neither', TypeError('test_name1 must be an instance of float, not int.')), (None, 'test_name1', numbers.Real, 2, 4, 'neither', TypeError('test_name1 must be an instance of float, not NoneType.')), (None, 'test_name1', numbers.Integral, 2, 4, 'neither', TypeError('test_name1 must be an instance of int, not NoneType.')), (1, 'test_name1', (float, bool), 2, 4, 'neither', TypeError('test_name1 must be an instance of {float, bool}, not int.')), (1, 'test_name2', int, 2, 4, 'neither', ValueError('test_name2 == 1, must be > 2.')), (5, 'test_name3', int, 2, 4, 'neither', ValueError('test_name3 == 5, must be < 4.')), (2, 'test_name4', int, 2, 4, 'right', ValueError('test_name4 == 2, must be > 2.')), (4, 'test_name5', int, 2, 4, 'left', ValueError('test_name5 == 4, must be < 4.')), (4, 'test_name6', int, 2, 4, 'bad parameter value', ValueError("Unknown value for `include_boundaries`: 'bad parameter value'. Possible values are: ('left', 'right', 'both', 'neither').")), (4, 'test_name7', int, None, 4, 'left', ValueError("`include_boundaries`='left' without specifying explicitly `min_val` is inconsistent.")), (4, 'test_name8', int, 2, None, 'right', ValueError("`include_boundaries`='right' without specifying explicitly `max_val` is inconsistent."))])
def test_check_scalar_invalid(x, target_name, target_type, min_val, max_val, include_boundaries, err_msg):
    """Test that check_scalar returns the right error if a wrong input is
    given"""
    with pytest.raises(Exception) as raised_error:
        check_scalar(x, target_name, target_type=target_type, min_val=min_val, max_val=max_val, include_boundaries=include_boundaries)
    assert str(raised_error.value) == str(err_msg)
    assert type(raised_error.value) == type(err_msg)