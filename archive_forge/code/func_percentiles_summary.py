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
def percentiles_summary(df, num_old, num_new, upsample, state):
    """Summarize data using percentiles and derived weights.

    These summaries can be merged, compressed, and converted back into
    approximate percentiles.

    Parameters
    ----------
    df: pandas.Series
        Data to summarize
    num_old: int
        Number of partitions of the current object
    num_new: int
        Number of partitions of the new object
    upsample: float
        Scale factor to increase the number of percentiles calculated in
        each partition.  Use to improve accuracy.
    """
    from dask.array.dispatch import percentile_lookup as _percentile
    from dask.array.utils import array_safe
    length = len(df)
    if length == 0:
        return ()
    random_state = np.random.RandomState(state)
    qs = sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df
    interpolation = 'linear'
    if isinstance(data.dtype, pd.CategoricalDtype):
        data = data.cat.codes
        interpolation = 'nearest'
    elif is_datetime64_dtype(data.dtype) or is_integer_dtype(data.dtype):
        interpolation = 'nearest'
    try:
        vals = data.quantile(q=qs / 100, interpolation=interpolation).values
    except (TypeError, NotImplementedError):
        try:
            vals, _ = _percentile(array_safe(data, like=data.values), qs, interpolation)
        except (TypeError, NotImplementedError):
            interpolation = 'nearest'
            vals = data.to_frame().quantile(q=qs / 100, interpolation=interpolation, numeric_only=False, method='table').iloc[:, 0]
    if is_cupy_type(data) and interpolation == 'linear' and np.issubdtype(data.dtype, np.integer):
        vals = np.round(vals).astype(data.dtype)
        if qs[0] == 0:
            vals[0] = data.min()
    vals_and_weights = percentiles_to_weights(qs, vals, length)
    return vals_and_weights