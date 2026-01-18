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
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)
    X_zero_samples = np.empty(0).reshape(0, 3)
    err_msg = f'The estimator {name} does not raise a ValueError when an empty data is used to train. Perhaps use check_array in train.'
    with raises(ValueError, err_msg=err_msg):
        e.fit(X_zero_samples, [])
    X_zero_features = np.empty(0).reshape(12, 0)
    y = _enforce_estimator_tags_y(e, np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    msg = '0 feature\\(s\\) \\(shape=\\(\\d*, 0\\)\\) while a minimum of \\d* is required.'
    with raises(ValueError, match=msg):
        e.fit(X_zero_features, y)