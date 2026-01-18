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
def check_classifier_multioutput(name, estimator):
    n_samples, n_labels, n_classes = (42, 5, 3)
    tags = _safe_tags(estimator)
    estimator = clone(estimator)
    X, y = make_multilabel_classification(random_state=42, n_samples=n_samples, n_labels=n_labels, n_classes=n_classes)
    estimator.fit(X, y)
    y_pred = estimator.predict(X)
    assert y_pred.shape == (n_samples, n_classes), 'The shape of the prediction for multioutput data is incorrect. Expected {}, got {}.'.format((n_samples, n_labels), y_pred.shape)
    assert y_pred.dtype.kind == 'i'
    if hasattr(estimator, 'decision_function'):
        decision = estimator.decision_function(X)
        assert isinstance(decision, np.ndarray)
        assert decision.shape == (n_samples, n_classes), 'The shape of the decision function output for multioutput data is incorrect. Expected {}, got {}.'.format((n_samples, n_classes), decision.shape)
        dec_pred = (decision > 0).astype(int)
        dec_exp = estimator.classes_[dec_pred]
        assert_array_equal(dec_exp, y_pred)
    if hasattr(estimator, 'predict_proba'):
        y_prob = estimator.predict_proba(X)
        if isinstance(y_prob, list) and (not tags['poor_score']):
            for i in range(n_classes):
                assert y_prob[i].shape == (n_samples, 2), 'The shape of the probability for multioutput data is incorrect. Expected {}, got {}.'.format((n_samples, 2), y_prob[i].shape)
                assert_array_equal(np.argmax(y_prob[i], axis=1).astype(int), y_pred[:, i])
        elif not tags['poor_score']:
            assert y_prob.shape == (n_samples, n_classes), 'The shape of the probability for multioutput data is incorrect. Expected {}, got {}.'.format((n_samples, n_classes), y_prob.shape)
            assert_array_equal(y_prob.round().astype(int), y_pred)
    if hasattr(estimator, 'decision_function') and hasattr(estimator, 'predict_proba'):
        for i in range(n_classes):
            y_proba = estimator.predict_proba(X)[:, i]
            y_decision = estimator.decision_function(X)
            assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))