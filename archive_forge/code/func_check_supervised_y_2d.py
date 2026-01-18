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
def check_supervised_y_2d(name, estimator_orig):
    tags = _safe_tags(estimator_orig)
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _enforce_estimator_tags_X(estimator_orig, rnd.uniform(size=(n_samples, 3)))
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator_orig, y)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    set_random_state(estimator)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DataConversionWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = 'expected 1 DataConversionWarning, got: %s' % ', '.join([str(w_x) for w_x in w])
    if not tags['multioutput']:
        assert len(w) > 0, msg
        assert "DataConversionWarning('A column-vector y was passed when a 1d array was expected" in msg
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())