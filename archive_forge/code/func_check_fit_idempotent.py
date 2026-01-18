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
def check_fit_idempotent(name, estimator_orig):
    check_methods = ['predict', 'transform', 'decision_function', 'predict_proba']
    rng = np.random.RandomState(0)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params().keys():
        estimator.set_params(warm_start=False)
    n_samples = 100
    X = rng.normal(loc=100, size=(n_samples, 2))
    X = _enforce_estimator_tags_X(estimator, X)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)
    y = _enforce_estimator_tags_y(estimator, y)
    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    result = {method: getattr(estimator, method)(X_test) for method in check_methods if hasattr(estimator, method)}
    set_random_state(estimator)
    estimator.fit(X_train, y_train)
    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2 * np.finfo(new_result.dtype).eps
            else:
                tol = 2 * np.finfo(np.float64).eps
            assert_allclose_dense_sparse(result[method], new_result, atol=max(tol, 1e-09), rtol=max(tol, 1e-07), err_msg='Idempotency check failed for method {}'.format(method))