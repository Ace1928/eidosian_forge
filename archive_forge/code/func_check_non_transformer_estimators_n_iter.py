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
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    not_run_check_n_iter = ['Ridge', 'RidgeClassifier', 'RandomizedLasso', 'LogisticRegressionCV', 'LinearSVC', 'LogisticRegression', 'SelfTrainingClassifier']
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.0)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        iris = load_iris()
        X, y_ = (iris.data, iris.target)
        y_ = _enforce_estimator_tags_y(estimator, y_)
        set_random_state(estimator, 0)
        X = _enforce_estimator_tags_X(estimator_orig, X)
        estimator.fit(X, y_)
        assert np.all(estimator.n_iter_ >= 1)