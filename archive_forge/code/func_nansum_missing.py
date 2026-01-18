from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def nansum_missing(array, axis):
    """nansum where all-NaN values remain NaNs.

    Note: In NumPy <=1.9 NaN is returned for slices that are
    all NaN, while later versions return 0. This function emulates
    the older behavior, which allows using NaN as a missing value
    indicator.

    Parameters
    ----------
    array: Array to sum over
    axis:  Axis to sum over
    """
    T = list(range(array.ndim))
    T.remove(axis)
    T.insert(0, axis)
    array = array.transpose(T)
    missing_vals = np.isnan(array)
    all_empty = np.all(missing_vals, axis=0)
    set_to_zero = missing_vals & ~all_empty
    return np.where(set_to_zero, 0, array).sum(axis=0)