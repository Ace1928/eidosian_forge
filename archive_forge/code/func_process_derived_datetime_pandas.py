import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def process_derived_datetime_pandas(data, not_found, indexes=None):
    from pandas.api.types import is_datetime64_any_dtype as isdate
    indexes = indexes or []
    extra_cols = {}
    for var in not_found:
        if '.' in var:
            parts = var.split('.')
            base_col = parts[0]
            dt_str = parts[-1]
            if base_col in data.columns:
                if isdate(data[base_col]):
                    extra_cols[var] = getattr(data[base_col].dt, dt_str)
            elif base_col == 'index':
                if isdate(data.index):
                    extra_cols[var] = getattr(data.index, dt_str)
            elif base_col in indexes:
                index = data.axes[indexes.index(base_col)]
                if isdate(index):
                    extra_cols[var] = getattr(index, dt_str)
    if extra_cols:
        data = data.assign(**extra_cols)
    not_found = [var for var in not_found if var not in extra_cols.keys()]
    return (not_found, data)