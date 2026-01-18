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
def test_check_memory():
    memory = check_memory('cache_directory')
    assert memory.location == 'cache_directory'
    memory = check_memory(None)
    assert memory.location is None
    dummy = DummyMemory()
    memory = check_memory(dummy)
    assert memory is dummy
    msg = "'memory' should be None, a string or have the same interface as joblib.Memory. Got memory='1' instead."
    with pytest.raises(ValueError, match=msg):
        check_memory(1)
    dummy = WrongDummyMemory()
    msg = "'memory' should be None, a string or have the same interface as joblib.Memory. Got memory='{}' instead.".format(dummy)
    with pytest.raises(ValueError, match=msg):
        check_memory(dummy)