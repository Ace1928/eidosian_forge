from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype
from tlz import merge, merge_sorted, take
from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.dispatch import tolist_dispatch
from dask.utils import is_cupy_type, random_state_data
def percentiles_to_weights(qs, vals, length):
    """Weigh percentile values by length and the difference between percentiles

    >>> percentiles = np.array([0., 25., 50., 90., 100.])
    >>> values = np.array([2, 3, 5, 8, 13])
    >>> length = 10
    >>> percentiles_to_weights(percentiles, values, length)
    ([2, 3, 5, 8, 13], [125.0, 250.0, 325.0, 250.0, 50.0])

    The weight of the first element, ``2``, is determined by the difference
    between the first and second percentiles, and then scaled by length:

    >>> 0.5 * length * (percentiles[1] - percentiles[0])
    125.0

    The second weight uses the difference of percentiles on both sides, so
    it will be twice the first weight if the percentiles are equally spaced:

    >>> 0.5 * length * (percentiles[2] - percentiles[0])
    250.0
    """
    if length == 0:
        return ()
    diff = np.ediff1d(qs, 0.0, 0.0)
    weights = 0.5 * length * (diff[1:] + diff[:-1])
    try:
        return (tolist_dispatch(vals), weights.tolist())
    except TypeError:
        return (vals.tolist(), weights.tolist())