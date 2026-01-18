from __future__ import annotations
import itertools
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar
def is_range_like(obj):
    """
    Check if the object is range-like.

    Objects that are considered range-like have information about the range (start and
    stop positions, and step) and also have to be iterable. Examples of range-like
    objects are: Python range, pandas.RangeIndex.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return hasattr(obj, '__iter__') and hasattr(obj, 'start') and hasattr(obj, 'stop') and hasattr(obj, 'step')