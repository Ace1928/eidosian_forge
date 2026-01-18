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
def check_classifiers_multilabel_output_format_predict_proba(name, classifier_orig):
    """Check the output of the `predict_proba` method for classifiers supporting
    multilabel-indicator targets."""
    classifier = clone(classifier_orig)
    set_random_state(classifier)
    n_samples, test_size, n_outputs = (100, 25, 5)
    X, y = make_multilabel_classification(n_samples=n_samples, n_features=2, n_classes=n_outputs, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    X = scale(X)
    X_train, X_test = (X[:-test_size], X[-test_size:])
    y_train = y[:-test_size]
    classifier.fit(X_train, y_train)
    response_method_name = 'predict_proba'
    predict_proba_method = getattr(classifier, response_method_name, None)
    if predict_proba_method is None:
        raise SkipTest(f'{name} does not have a {response_method_name} method.')
    y_pred = predict_proba_method(X_test)
    if isinstance(y_pred, list):
        assert len(y_pred) == n_outputs, f'When {name}.predict_proba returns a list, the list should be of length n_outputs and contain NumPy arrays. Got length of {len(y_pred)} instead of {n_outputs}.'
        for pred in y_pred:
            assert pred.shape == (test_size, 2), f'When {name}.predict_proba returns a list, this list should contain NumPy arrays of shape (n_samples, 2). Got NumPy arrays of shape {pred.shape} instead of {(test_size, 2)}.'
            assert pred.dtype.kind == 'f', f'When {name}.predict_proba returns a list, it should contain NumPy arrays with floating dtype. Got {pred.dtype} instead.'
            err_msg = f'When {name}.predict_proba returns a list, each NumPy array should contain probabilities for each class and thus each row should sum to 1 (or close to 1 due to numerical errors).'
            assert_allclose(pred.sum(axis=1), 1, err_msg=err_msg)
    elif isinstance(y_pred, np.ndarray):
        assert y_pred.shape == (test_size, n_outputs), f'When {name}.predict_proba returns a NumPy array, the expected shape is (n_samples, n_outputs). Got {y_pred.shape} instead of {(test_size, n_outputs)}.'
        assert y_pred.dtype.kind == 'f', f'When {name}.predict_proba returns a NumPy array, the expected data type is floating. Got {y_pred.dtype} instead.'
        err_msg = f'When {name}.predict_proba returns a NumPy array, this array is expected to provide probabilities of the positive class and should therefore contain values between 0 and 1.'
        assert_array_less(0, y_pred, err_msg=err_msg)
        assert_array_less(y_pred, 1, err_msg=err_msg)
    else:
        raise ValueError(f'Unknown returned type {type(y_pred)} by {name}.predict_proba. A list or a Numpy array is expected.')