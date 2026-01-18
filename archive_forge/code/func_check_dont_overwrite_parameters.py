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
def check_dont_overwrite_parameters(name, estimator_orig):
    if hasattr(estimator_orig.__init__, 'deprecated_original'):
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, 'n_components'):
        estimator.n_components = 1
    if hasattr(estimator, 'n_clusters'):
        estimator.n_clusters = 1
    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)
    dict_after_fit = estimator.__dict__
    public_keys_after_fit = [key for key in dict_after_fit.keys() if _is_public_parameter(key)]
    attrs_added_by_fit = [key for key in public_keys_after_fit if key not in dict_before_fit.keys()]
    assert not attrs_added_by_fit, 'Estimator adds public attribute(s) during the fit method. Estimators are only allowed to add private attributes either started with _ or ended with _ but %s added' % ', '.join(attrs_added_by_fit)
    attrs_changed_by_fit = [key for key in public_keys_after_fit if dict_before_fit[key] is not dict_after_fit[key]]
    assert not attrs_changed_by_fit, 'Estimator changes public attribute(s) during the fit method. Estimators are only allowed to change attributes started or ended with _, but %s changed' % ', '.join(attrs_changed_by_fit)