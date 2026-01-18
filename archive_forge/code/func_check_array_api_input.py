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
def check_array_api_input(name, estimator_orig, array_namespace, device=None, dtype_name='float64', check_values=False):
    """Check that the estimator can work consistently with the Array API

    By default, this just checks that the types and shapes of the arrays are
    consistent with calling the same estimator with numpy arrays.

    When check_values is True, it also checks that calling the estimator on the
    array_api Array gives the same results as ndarrays.
    """
    xp = _array_api_for_tests(array_namespace, device)
    X, y = make_classification(random_state=42)
    X = X.astype(dtype_name, copy=False)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    est = clone(estimator_orig)
    X_xp = xp.asarray(X, device=device)
    y_xp = xp.asarray(y, device=device)
    est.fit(X, y)
    array_attributes = {key: value for key, value in vars(est).items() if isinstance(value, np.ndarray)}
    est_xp = clone(est)
    with config_context(array_api_dispatch=True):
        est_xp.fit(X_xp, y_xp)
        input_ns = get_namespace(X_xp)[0].__name__
    for key, attribute in array_attributes.items():
        est_xp_param = getattr(est_xp, key)
        with config_context(array_api_dispatch=True):
            attribute_ns = get_namespace(est_xp_param)[0].__name__
        assert attribute_ns == input_ns, f"'{key}' attribute is in wrong namespace, expected {input_ns} got {attribute_ns}"
        assert array_device(est_xp_param) == array_device(X_xp)
        est_xp_param_np = _convert_to_numpy(est_xp_param, xp=xp)
        if check_values:
            assert_allclose(attribute, est_xp_param_np, err_msg=f'{key} not the same', atol=np.finfo(X.dtype).eps * 100)
        else:
            assert attribute.shape == est_xp_param_np.shape
            assert attribute.dtype == est_xp_param_np.dtype
    methods = ('score', 'score_samples', 'decision_function', 'predict', 'predict_log_proba', 'predict_proba', 'transform')
    for method_name in methods:
        method = getattr(est, method_name, None)
        if method is None:
            continue
        if method_name == 'score':
            result = method(X, y)
            with config_context(array_api_dispatch=True):
                result_xp = getattr(est_xp, method_name)(X_xp, y_xp)
            assert isinstance(result, float)
            assert isinstance(result_xp, float)
            if check_values:
                assert abs(result - result_xp) < np.finfo(X.dtype).eps * 100
            continue
        else:
            result = method(X)
            with config_context(array_api_dispatch=True):
                result_xp = getattr(est_xp, method_name)(X_xp)
        with config_context(array_api_dispatch=True):
            result_ns = get_namespace(result_xp)[0].__name__
        assert result_ns == input_ns, f"'{method}' output is in wrong namespace, expected {input_ns}, got {result_ns}."
        assert array_device(result_xp) == array_device(X_xp)
        result_xp_np = _convert_to_numpy(result_xp, xp=xp)
        if check_values:
            assert_allclose(result, result_xp_np, err_msg=f'{method} did not the return the same result', atol=np.finfo(X.dtype).eps * 100)
        elif hasattr(result, 'shape'):
            assert result.shape == result_xp_np.shape
            assert result.dtype == result_xp_np.dtype
        if method_name == 'transform' and hasattr(est, 'inverse_transform'):
            inverse_result = est.inverse_transform(result)
            with config_context(array_api_dispatch=True):
                invese_result_xp = est_xp.inverse_transform(result_xp)
                inverse_result_ns = get_namespace(invese_result_xp)[0].__name__
            assert inverse_result_ns == input_ns, f"'inverse_transform' output is in wrong namespace, expected {input_ns}, got {inverse_result_ns}."
            assert array_device(invese_result_xp) == array_device(X_xp)
            invese_result_xp_np = _convert_to_numpy(invese_result_xp, xp=xp)
            if check_values:
                assert_allclose(inverse_result, invese_result_xp_np, err_msg='inverse_transform did not the return the same result', atol=np.finfo(X.dtype).eps * 100)
            else:
                assert inverse_result.shape == invese_result_xp_np.shape
                assert inverse_result.dtype == invese_result_xp_np.dtype