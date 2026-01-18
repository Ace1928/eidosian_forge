from __future__ import annotations
import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.errors import PerformanceWarning
from tlz import partition
from dask.dataframe._compat import (
from dask.dataframe.dispatch import (  # noqa: F401
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like
from dask.utils import _deprecated_kwarg
def describe_aggregate(values):
    assert len(values) > 0
    names = []
    values_indexes = sorted((x.index for x in values), key=len)
    for idxnames in values_indexes:
        for name in idxnames:
            if name not in names:
                names.append(name)
    return pd.concat(values, axis=1, sort=False).reindex(names)