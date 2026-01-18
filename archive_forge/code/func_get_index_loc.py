from __future__ import annotations
from statsmodels.compat.pandas import (
import numbers
import warnings
import numpy as np
from pandas import (
from pandas.tseries.frequencies import to_offset
from statsmodels.base.data import PandasData
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ValueWarning
def get_index_loc(key, index):
    """
    Get the location of a specific key in an index

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or a location if the underlying index is a RangeIndex
        or an Index with an integer dtype.
    index : pd.Index
        The index to search.

    Returns
    -------
    loc : int
        The location of the key
    index : pd.Index
        The index including the key; this is a copy of the original index
        unless the index had to be expanded to accommodate `key`.
    index_was_expanded : bool
        Whether or not the index was expanded to accommodate `key`.

    Notes
    -----
    If `key` is past the end of of the given index, and the index is either
    an Index with an integral dtype or a date index, this function extends
    the index up to and including key, and then returns the location in the
    new index.
    """
    base_index = index
    index = base_index
    date_index = isinstance(base_index, (PeriodIndex, DatetimeIndex))
    int_index = is_int_index(base_index)
    range_index = isinstance(base_index, RangeIndex)
    index_class = type(base_index)
    nobs = len(index)
    if range_index and isinstance(key, (int, np.integer)):
        if key < 0 and -key <= nobs:
            key = nobs + key
        elif key > nobs - 1:
            try:
                base_index_start = base_index.start
                base_index_step = base_index.step
            except AttributeError:
                base_index_start = base_index._start
                base_index_step = base_index._step
            stop = base_index_start + (key + 1) * base_index_step
            index = RangeIndex(start=base_index_start, stop=stop, step=base_index_step)
    if not range_index and int_index and (not date_index) and isinstance(key, (int, np.integer)):
        if key < 0 and -key <= nobs:
            key = nobs + key
        elif key > base_index[-1]:
            index = Index(np.arange(base_index[0], int(key + 1)))
    if date_index:
        if index_class is DatetimeIndex:
            index_fn = date_range
        else:
            index_fn = period_range
        if isinstance(key, (int, np.integer)):
            if key < 0 and -key < nobs:
                key = index[nobs + key]
            elif key > len(base_index) - 1:
                index = index_fn(start=base_index[0], periods=int(key + 1), freq=base_index.freq)
                key = index[-1]
            else:
                key = index[key]
        else:
            if index_class is PeriodIndex:
                date_key = Period(key, freq=base_index.freq)
            else:
                date_key = Timestamp(key)
            if date_key > base_index[-1]:
                index = index_fn(start=base_index[0], end=date_key, freq=base_index.freq)
                if not index[-1] == date_key:
                    index = index_fn(start=base_index[0], periods=len(index) + 1, freq=base_index.freq)
                key = index[-1]
    if date_index:
        loc = index.get_loc(key)
    elif int_index or range_index:
        try:
            index[key]
        except (IndexError, ValueError) as e:
            raise KeyError(str(e))
        loc = key
    else:
        loc = index.get_loc(key)
    index_was_expanded = index is not base_index
    if isinstance(loc, slice):
        end = loc.stop - 1
    else:
        end = loc
    return (loc, index[:end + 1], index_was_expanded)