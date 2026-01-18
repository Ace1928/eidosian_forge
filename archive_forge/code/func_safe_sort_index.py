from __future__ import annotations
import textwrap
from typing import (
import numpy as np
from pandas._libs import (
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.cast import find_common_type
from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
def safe_sort_index(index: Index) -> Index:
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    if index.is_monotonic_increasing:
        return index
    try:
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        if isinstance(array_sorted, Index):
            return array_sorted
        array_sorted = cast(np.ndarray, array_sorted)
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            index = Index(array_sorted, name=index.name, dtype=index.dtype)
    return index