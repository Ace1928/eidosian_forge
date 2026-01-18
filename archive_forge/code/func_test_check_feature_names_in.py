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
def test_check_feature_names_in():
    """Check behavior of check_feature_names_in for arrays."""
    X = np.array([[0.0, 1.0, 2.0]])
    est = PassthroughTransformer().fit(X)
    names = est.get_feature_names_out()
    assert_array_equal(names, ['x0', 'x1', 'x2'])
    incorrect_len_names = ['x10', 'x1']
    with pytest.raises(ValueError, match='input_features should have length equal to'):
        est.get_feature_names_out(incorrect_len_names)
    del est.n_features_in_
    with pytest.raises(ValueError, match='Unable to generate feature names'):
        est.get_feature_names_out()