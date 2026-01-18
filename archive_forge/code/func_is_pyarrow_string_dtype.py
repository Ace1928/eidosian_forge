from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def is_pyarrow_string_dtype(dtype):
    """Is the input dtype a pyarrow string?"""
    if pa is None:
        return False
    if PANDAS_GE_150:
        pa_string_types = [pd.StringDtype('pyarrow'), pd.ArrowDtype(pa.string())]
    else:
        pa_string_types = [pd.StringDtype('pyarrow')]
    return dtype in pa_string_types