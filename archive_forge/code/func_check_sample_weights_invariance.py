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
def check_sample_weights_invariance(name, estimator_orig, kind='ones'):
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)
    X1 = np.array([[1, 3], [1, 3], [1, 3], [1, 3], [2, 1], [2, 1], [2, 1], [2, 1], [3, 3], [3, 3], [3, 3], [3, 3], [4, 1], [4, 1], [4, 1], [4, 1]], dtype=np.float64)
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    if kind == 'ones':
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        err_msg = f'For {name} sample_weight=None is not equivalent to sample_weight=ones'
    elif kind == 'zeros':
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1):] = 0
        X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)
        err_msg = f'For {name}, a zero sample_weight is not equivalent to removing the sample'
    else:
        raise ValueError
    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)
    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)
    for method in ['predict', 'predict_proba', 'decision_function', 'transform']:
        if hasattr(estimator_orig, method):
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)