import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
@ignore_warnings(category=FutureWarning)
def check_class_weight_classifiers(name, classifier_orig):
    if _safe_tags(classifier_orig, key='binary_only'):
        problems = [2]
    else:
        problems = [2, 3]
    for n_centers in problems:
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        if _safe_tags(classifier_orig, key='pairwise'):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)
        n_centers = len(np.unique(y_train))
        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}
        classifier = clone(classifier_orig).set_params(class_weight=class_weight)
        if hasattr(classifier, 'n_iter'):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, 'max_iter'):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, 'min_weight_fraction_leaf'):
            classifier.set_params(min_weight_fraction_leaf=0.01)
        if hasattr(classifier, 'n_iter_no_change'):
            classifier.set_params(n_iter_no_change=20)
        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        if not _safe_tags(classifier_orig, key='poor_score'):
            assert np.mean(y_pred == 0) > 0.87