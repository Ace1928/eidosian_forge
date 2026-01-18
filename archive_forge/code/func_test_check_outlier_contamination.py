import importlib
import sys
import unittest
import warnings
from numbers import Integral, Real
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn import config_context, get_config
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning, SkipTestWarning
from sklearn.linear_model import (
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, NuSVC
from sklearn.utils import _array_api, all_estimators, deprecated
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
def test_check_outlier_contamination():
    """Check the test for the contamination parameter in the outlier detectors."""

    class OutlierDetectorWithoutConstraint(OutlierMixin, BaseEstimator):
        """Outlier detector without parameter validation."""

        def __init__(self, contamination=0.1):
            self.contamination = contamination

        def fit(self, X, y=None, sample_weight=None):
            return self

        def predict(self, X, y=None):
            return np.ones(X.shape[0])
    detector = OutlierDetectorWithoutConstraint()
    assert check_outlier_contamination(detector.__class__.__name__, detector) is None

    class OutlierDetectorWithConstraint(OutlierDetectorWithoutConstraint):
        _parameter_constraints = {'contamination': [StrOptions({'auto'})]}
    detector = OutlierDetectorWithConstraint()
    err_msg = 'contamination constraints should contain a Real Interval constraint.'
    with raises(AssertionError, match=err_msg):
        check_outlier_contamination(detector.__class__.__name__, detector)
    OutlierDetectorWithConstraint._parameter_constraints['contamination'] = [Interval(Real, 0, 0.5, closed='right')]
    detector = OutlierDetectorWithConstraint()
    check_outlier_contamination(detector.__class__.__name__, detector)
    incorrect_intervals = [Interval(Integral, 0, 1, closed='right'), Interval(Real, -1, 1, closed='right'), Interval(Real, 0, 2, closed='right'), Interval(Real, 0, 0.5, closed='left')]
    err_msg = 'contamination constraint should be an interval in \\(0, 0.5\\]'
    for interval in incorrect_intervals:
        OutlierDetectorWithConstraint._parameter_constraints['contamination'] = [interval]
        detector = OutlierDetectorWithConstraint()
        with raises(AssertionError, match=err_msg):
            check_outlier_contamination(detector.__class__.__name__, detector)