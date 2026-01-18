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
def get_scorer_names():
    """Get the names of all available scorers.

    These names can be passed to :func:`~sklearn.metrics.get_scorer` to
    retrieve the scorer object.

    Returns
    -------
    list of str
        Names of all available scorers.

    Examples
    --------
    >>> from sklearn.metrics import get_scorer_names
    >>> all_scorers = get_scorer_names()
    >>> type(all_scorers)
    <class 'list'>
    >>> all_scorers[:3]
    ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score']
    >>> "roc_auc" in all_scorers
    True
    """
    return sorted(_SCORERS.keys())