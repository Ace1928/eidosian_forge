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
def test_check_fit_check_is_fitted():

    class Estimator(BaseEstimator):

        def __init__(self, behavior='attribute'):
            self.behavior = behavior

        def fit(self, X, y, **kwargs):
            if self.behavior == 'attribute':
                self.is_fitted_ = True
            elif self.behavior == 'method':
                self._is_fitted = True
            return self

        @available_if(lambda self: self.behavior in {'method', 'always-true'})
        def __sklearn_is_fitted__(self):
            if self.behavior == 'always-true':
                return True
            return hasattr(self, '_is_fitted')
    with raises(Exception, match='passes check_is_fitted before being fit'):
        check_fit_check_is_fitted('estimator', Estimator(behavior='always-true'))
    check_fit_check_is_fitted('estimator', Estimator(behavior='method'))
    check_fit_check_is_fitted('estimator', Estimator(behavior='attribute'))