from __future__ import annotations
from functools import partial
import pandas as pd
from packaging.version import Version
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
def is_object_string_dataframe(x):
    return isinstance(x, pd.DataFrame) and (any((is_object_string_series(s) for _, s in x.items())) or is_object_string_index(x.index))