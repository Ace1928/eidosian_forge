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
@validate_params({'scoring': [str, callable, None]}, prefer_skip_nested_validation=True)
def get_scorer(scoring):
    """Get a scorer from string.

    Read more in the :ref:`User Guide <scoring_parameter>`.
    :func:`~sklearn.metrics.get_scorer_names` can be used to retrieve the names
    of all available scorers.

    Parameters
    ----------
    scoring : str, callable or None
        Scoring method as string. If callable it is returned as is.
        If None, returns None.

    Returns
    -------
    scorer : callable
        The scorer.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer
    object. Calling `get_scorer` twice for the same scorer results in two
    separate scorer objects.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.metrics import get_scorer
    >>> X = np.reshape([0, 1, -1, -0.5, 2], (-1, 1))
    >>> y = np.array([0, 1, 1, 0, 1])
    >>> classifier = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    >>> accuracy = get_scorer("accuracy")
    >>> accuracy(classifier, X, y)
    0.4
    """
    if isinstance(scoring, str):
        try:
            scorer = copy.deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError('%r is not a valid scoring value. Use sklearn.metrics.get_scorer_names() to get valid options.' % scoring)
    else:
        scorer = scoring
    return scorer