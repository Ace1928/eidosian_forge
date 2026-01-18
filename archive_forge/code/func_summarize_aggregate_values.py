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
def summarize_aggregate_values(aggregate, how='linear', num=180):
    """Helper function similar to np.linspace which return values from aggregate min value to
    aggregate max value in either linear or log space.
    """
    max_val = np.nanmax(aggregate.values)
    min_val = np.nanmin(aggregate.values)
    if min_val == 0:
        min_val = aggregate.data[aggregate.data > 0].min()
    if how == 'linear':
        vals = np.linspace(min_val, max_val, num)[None, :]
    else:
        vals = (np.logspace(0, np.log1p(max_val - min_val), base=np.e, num=num, dtype=min_val.dtype) + min_val)[None, :]
    return (DataArray(vals), min_val, max_val)