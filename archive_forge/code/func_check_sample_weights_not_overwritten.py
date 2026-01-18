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
def check_sample_weights_not_overwritten(name, estimator_orig):
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)
    X = np.array([[1, 3], [1, 3], [1, 3], [1, 3], [2, 1], [2, 1], [2, 1], [2, 1], [3, 3], [3, 3], [3, 3], [3, 3], [4, 1], [4, 1], [4, 1], [4, 1]], dtype=np.float64)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    y = _enforce_estimator_tags_y(estimator, y)
    sample_weight_original = np.ones(y.shape[0])
    sample_weight_original[0] = 10.0
    sample_weight_fit = sample_weight_original.copy()
    estimator.fit(X, y, sample_weight=sample_weight_fit)
    err_msg = f'{name} overwrote the original `sample_weight` given during fit'
    assert_allclose(sample_weight_fit, sample_weight_original, err_msg=err_msg)