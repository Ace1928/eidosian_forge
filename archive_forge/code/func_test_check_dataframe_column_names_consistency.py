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
def test_check_dataframe_column_names_consistency():
    err_msg = 'Estimator does not have a feature_names_in_'
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency('estimator_name', BaseBadClassifier())
    check_dataframe_column_names_consistency('estimator_name', PartialFitChecksName())
    lr = LogisticRegression()
    check_dataframe_column_names_consistency(lr.__class__.__name__, lr)
    lr.__doc__ = "Docstring that does not document the estimator's attributes"
    err_msg = 'Estimator LogisticRegression does not document its feature_names_in_ attribute'
    with raises(ValueError, match=err_msg):
        check_dataframe_column_names_consistency(lr.__class__.__name__, lr)