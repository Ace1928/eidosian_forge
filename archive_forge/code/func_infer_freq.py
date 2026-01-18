from __future__ import annotations
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import _MONTH_ABBREVIATIONS, _legacy_to_new_freq
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.common import _contains_datetime_like_objects
def infer_freq(index):
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : CFTimeIndex, DataArray, DatetimeIndex, TimedeltaIndex, Series
        If not passed a CFTimeIndex, this simply calls `pandas.infer_freq`.
        If passed a Series or a DataArray will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values or the index is not 1D.
    """
    from xarray.core.dataarray import DataArray
    from xarray.core.variable import Variable
    if isinstance(index, (DataArray, pd.Series)):
        if index.ndim != 1:
            raise ValueError("'index' must be 1D")
        elif not _contains_datetime_like_objects(Variable('dim', index)):
            raise ValueError("'index' must contain datetime-like objects")
        dtype = np.asarray(index).dtype
        if dtype == 'datetime64[ns]':
            index = pd.DatetimeIndex(index.values)
        elif dtype == 'timedelta64[ns]':
            index = pd.TimedeltaIndex(index.values)
        else:
            index = CFTimeIndex(index.values)
    if isinstance(index, CFTimeIndex):
        inferer = _CFTimeFrequencyInferer(index)
        return inferer.get_freq()
    return _legacy_to_new_freq(pd.infer_freq(index))