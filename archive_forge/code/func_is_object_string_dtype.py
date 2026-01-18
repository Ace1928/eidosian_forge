from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def is_object_string_dtype(dtype):
    """Determine if input is a non-pyarrow string dtype"""
    return pd.api.types.is_string_dtype(dtype) and (not is_pyarrow_string_dtype(dtype)) and (not pd.api.types.is_dtype_equal(dtype, 'decimal'))