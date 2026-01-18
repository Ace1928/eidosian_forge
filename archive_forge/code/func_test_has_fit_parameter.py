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
def test_has_fit_parameter():
    assert not has_fit_parameter(KNeighborsClassifier, 'sample_weight')
    assert has_fit_parameter(RandomForestRegressor, 'sample_weight')
    assert has_fit_parameter(SVR, 'sample_weight')
    assert has_fit_parameter(SVR(), 'sample_weight')

    class TestClassWithDeprecatedFitMethod:

        @deprecated('Deprecated for the purpose of testing has_fit_parameter')
        def fit(self, X, y, sample_weight=None):
            pass
    assert has_fit_parameter(TestClassWithDeprecatedFitMethod, 'sample_weight'), 'has_fit_parameter fails for class with deprecated fit method.'