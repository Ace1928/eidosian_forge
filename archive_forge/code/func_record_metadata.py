from functools import partial
import numpy as np
from sklearn.base import (
from sklearn.metrics._scorer import _Scorer, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.utils._metadata_requests import (
from sklearn.utils.metadata_routing import (
from sklearn.utils.multiclass import _check_partial_fit_first_call
def record_metadata(obj, method, record_default=True, **kwargs):
    """Utility function to store passed metadata to a method.

    If record_default is False, kwargs whose values are "default" are skipped.
    This is so that checks on keyword arguments whose default was not changed
    are skipped.

    """
    if not hasattr(obj, '_records'):
        obj._records = {}
    if not record_default:
        kwargs = {key: val for key, val in kwargs.items() if not isinstance(val, str) or val != 'default'}
    obj._records[method] = kwargs