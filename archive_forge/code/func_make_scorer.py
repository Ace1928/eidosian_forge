import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
@validate_params({'score_func': [callable], 'response_method': [None, list, tuple, StrOptions({'predict', 'predict_proba', 'decision_function'})], 'greater_is_better': ['boolean'], 'needs_proba': ['boolean', Hidden(StrOptions({'deprecated'}))], 'needs_threshold': ['boolean', Hidden(StrOptions({'deprecated'}))]}, prefer_skip_nested_validation=True)
def make_scorer(score_func, *, response_method=None, greater_is_better=True, needs_proba='deprecated', needs_threshold='deprecated', **kwargs):
    """Make a scorer from a performance metric or loss function.

    A scorer is a wrapper around an arbitrary metric or loss function that is called
    with the signature `scorer(estimator, X, y_true, **kwargs)`.

    It is accepted in all scikit-learn estimators or functions allowing a `scoring`
    parameter.

    The parameter `response_method` allows to specify which method of the estimator
    should be used to feed the scoring/loss function.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    response_method : {"predict_proba", "decision_function", "predict"} or             list/tuple of such str, default=None

        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`decision_function` or
        :term:`predict`). Possible choices are:

        - if `str`, it corresponds to the name to the method to return;
        - if a list or tuple of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.
        - if `None`, it is equivalent to `"predict"`.

        .. versionadded:: 1.4

    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    needs_proba : bool, default=False
        Whether `score_func` requires `predict_proba` to get probability
        estimates out of a classifier.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).

        .. deprecated:: 1.4
           `needs_proba` is deprecated in version 1.4 and will be removed in
           1.6. Use `response_method="predict_proba"` instead.

    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a `decision_function` or `predict_proba` method.

        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).

        For example `average_precision` or the area under the roc curve
        can not be computed using discrete predictions alone.

        .. deprecated:: 1.4
           `needs_threshold` is deprecated in version 1.4 and will be removed
           in 1.6. Use `response_method=("decision_function", "predict_proba")`
           instead to preserve the same behaviour.

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, response_method='predict', beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    response_method = _get_response_method(response_method, needs_threshold, needs_proba)
    sign = 1 if greater_is_better else -1
    return _Scorer(score_func, sign, kwargs, response_method)