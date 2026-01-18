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
def get_index_label_loc(key, index, row_labels):
    """
    Get the location of a specific key in an index or model row labels

    Parameters
    ----------
    key : label
        The key for which to find the location if the underlying index is
        a DateIndex or is only being used as row labels, or a location if
        the underlying index is a RangeIndex or a NumericIndex.
    index : pd.Index
        The index to search.
    row_labels : pd.Index
        Row labels to search if key not found in index

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
    This function expands on `get_index_loc` by first trying the given
    base index (or the model's index if the base index was not given) and
    then falling back to try again with the model row labels as the base
    index.
    """
    try:
        loc, index, index_was_expanded = get_index_loc(key, index)
    except KeyError as e:
        try:
            if not isinstance(key, (int, np.integer)):
                loc = row_labels.get_loc(key)
            else:
                raise
            if isinstance(loc, slice):
                loc = loc.start
            if isinstance(loc, np.ndarray):
                if loc.dtype == bool:
                    loc = np.argmax(loc)
                else:
                    loc = loc[0]
            if not isinstance(loc, numbers.Integral):
                raise
            index = row_labels[:loc + 1]
            index_was_expanded = False
        except:
            raise e
    return (loc, index, index_was_expanded)