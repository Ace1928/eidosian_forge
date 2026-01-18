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
def test_check_is_fitted():
    with pytest.raises(TypeError):
        check_is_fitted(ARDRegression)
    with pytest.raises(TypeError):
        check_is_fitted('SVR')
    ard = ARDRegression()
    svr = SVR()
    try:
        with pytest.raises(NotFittedError):
            check_is_fitted(ard)
        with pytest.raises(NotFittedError):
            check_is_fitted(svr)
    except ValueError:
        assert False, 'check_is_fitted failed with ValueError'
    msg = 'Random message %(name)s, %(name)s'
    match = 'Random message ARDRegression, ARDRegression'
    with pytest.raises(ValueError, match=match):
        check_is_fitted(ard, msg=msg)
    msg = 'Another message %(name)s, %(name)s'
    match = 'Another message SVR, SVR'
    with pytest.raises(AttributeError, match=match):
        check_is_fitted(svr, msg=msg)
    ard.fit(*make_blobs())
    svr.fit(*make_blobs())
    assert check_is_fitted(ard) is None
    assert check_is_fitted(svr) is None