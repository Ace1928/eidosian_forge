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
def dtype_info(df):
    info = None
    if isinstance(df.dtype, pd.CategoricalDtype):
        data = df.values
        info = (data.categories, data.ordered)
    return (df.dtype, info)