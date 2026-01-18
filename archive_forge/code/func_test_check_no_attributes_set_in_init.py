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
def test_check_no_attributes_set_in_init():

    class NonConformantEstimatorPrivateSet(BaseEstimator):

        def __init__(self):
            self.you_should_not_set_this_ = None

    class NonConformantEstimatorNoParamSet(BaseEstimator):

        def __init__(self, you_should_set_this_=None):
            pass

    class ConformantEstimatorClassAttribute(BaseEstimator):
        __metadata_request__fit = {'foo': True}
    msg = "Estimator estimator_name should not set any attribute apart from parameters during init. Found attributes \\['you_should_not_set_this_'\\]."
    with raises(AssertionError, match=msg):
        check_no_attributes_set_in_init('estimator_name', NonConformantEstimatorPrivateSet())
    msg = 'Estimator estimator_name should store all parameters as an attribute during init'
    with raises(AttributeError, match=msg):
        check_no_attributes_set_in_init('estimator_name', NonConformantEstimatorNoParamSet())
    check_no_attributes_set_in_init('estimator_name', ConformantEstimatorClassAttribute())
    with config_context(enable_metadata_routing=True):
        check_no_attributes_set_in_init('estimator_name', ConformantEstimatorClassAttribute().set_fit_request(foo=True))