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
def test_deprecate_positional_args_warns_for_function():

    @_deprecate_positional_args
    def f1(a, b, *, c=1, d=1):
        pass
    with pytest.warns(FutureWarning, match='Pass c=3 as keyword args'):
        f1(1, 2, 3)
    with pytest.warns(FutureWarning, match='Pass c=3, d=4 as keyword args'):
        f1(1, 2, 3, 4)

    @_deprecate_positional_args
    def f2(a=1, *, b=1, c=1, d=1):
        pass
    with pytest.warns(FutureWarning, match='Pass b=2 as keyword args'):
        f2(1, 2)

    @_deprecate_positional_args
    def f3(a, *, b, c=1, d=1):
        pass
    with pytest.warns(FutureWarning, match='Pass b=2 as keyword args'):
        f3(1, 2)