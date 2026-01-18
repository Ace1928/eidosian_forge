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
def check_transformer_get_feature_names_out(name, transformer_orig):
    tags = transformer_orig._get_tags()
    if '2darray' not in tags['X_types'] or tags['no_validation']:
        return
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]], random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    transformer = clone(transformer_orig)
    X = _enforce_estimator_tags_X(transformer, X)
    n_features = X.shape[1]
    set_random_state(transformer)
    y_ = y
    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[np.asarray(y), np.asarray(y)]
        y_[::2, 1] *= 2
    X_transform = transformer.fit_transform(X, y=y_)
    input_features = [f'feature{i}' for i in range(n_features)]
    with raises(ValueError, match='input_features should have length equal'):
        transformer.get_feature_names_out(input_features[::2])
    feature_names_out = transformer.get_feature_names_out(input_features)
    assert feature_names_out is not None
    assert isinstance(feature_names_out, np.ndarray)
    assert feature_names_out.dtype == object
    assert all((isinstance(name, str) for name in feature_names_out))
    if isinstance(X_transform, tuple):
        n_features_out = X_transform[0].shape[1]
    else:
        n_features_out = X_transform.shape[1]
    assert len(feature_names_out) == n_features_out, f'Expected {n_features_out} feature names, got {len(feature_names_out)}'