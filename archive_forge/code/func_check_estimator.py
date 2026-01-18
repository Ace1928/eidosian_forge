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
def check_estimator(estimator=None, generate_only=False):
    """Check if estimator adheres to scikit-learn conventions.

    This function will run an extensive test-suite for input validation,
    shapes, etc, making sure that the estimator complies with `scikit-learn`
    conventions as detailed in :ref:`rolling_your_own_estimator`.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    Setting `generate_only=True` returns a generator that yields (estimator,
    check) tuples where the check can be called independently from each
    other, i.e. `check(estimator)`. This allows all checks to be run
    independently and report the checks that are failing.

    scikit-learn provides a pytest specific decorator,
    :func:`~sklearn.utils.estimator_checks.parametrize_with_checks`, making it
    easier to test multiple estimators.

    Parameters
    ----------
    estimator : estimator object
        Estimator instance to check.

        .. versionadded:: 1.1
           Passing a class was deprecated in version 0.23, and support for
           classes was removed in 0.24.

    generate_only : bool, default=False
        When `False`, checks are evaluated when `check_estimator` is called.
        When `True`, `check_estimator` returns a generator that yields
        (estimator, check) tuples. The check is run by calling
        `check(estimator)`.

        .. versionadded:: 0.22

    Returns
    -------
    checks_generator : generator
        Generator that yields (estimator, check) tuples. Returned when
        `generate_only=True`.

    See Also
    --------
    parametrize_with_checks : Pytest specific decorator for parametrizing estimator
        checks.

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> check_estimator(LogisticRegression(), generate_only=True)
    <generator object ...>
    """
    if isinstance(estimator, type):
        msg = "Passing a class was deprecated in version 0.23 and isn't supported anymore from 0.24.Please pass an instance instead."
        raise TypeError(msg)
    name = type(estimator).__name__

    def checks_generator():
        for check in _yield_all_checks(estimator):
            check = _maybe_skip(estimator, check)
            yield (estimator, partial(check, name))
    if generate_only:
        return checks_generator()
    for estimator, check in checks_generator():
        try:
            check(estimator)
        except SkipTest as exception:
            warnings.warn(str(exception), SkipTestWarning)