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
@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = _enforce_estimator_tags_X(estimator_orig, rng.uniform(size=(40, 10)))
    X = X.astype(object)
    tags = _safe_tags(estimator_orig)
    y = (X[:, 0] * 4).astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    estimator.fit(X, y)
    if hasattr(estimator, 'predict'):
        estimator.predict(X)
    if hasattr(estimator, 'transform'):
        estimator.transform(X)
    with raises(Exception, match='Unknown label type', may_pass=True):
        estimator.fit(X, y.astype(object))
    if 'string' not in tags['X_types']:
        X[0, 0] = {'foo': 'bar'}
        msg = 'argument must be .* string.* number'
        with raises(TypeError, match=msg):
            estimator.fit(X, y)
    else:
        estimator.fit(X, y)