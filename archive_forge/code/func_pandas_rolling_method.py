from __future__ import annotations
import datetime
import warnings
from numbers import Integral
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.core.window import Rolling as pd_Rolling
from dask.array.core import normalize_arg
from dask.base import tokenize
from dask.blockwise import BlockwiseDepDict
from dask.dataframe import methods
from dask.dataframe._compat import check_axis_keyword_deprecation
from dask.dataframe.core import (
from dask.dataframe.io import from_pandas
from dask.dataframe.multi import _maybe_align_partitions
from dask.dataframe.utils import (
from dask.delayed import unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.typing import no_default
from dask.utils import M, apply, derived_from, funcname, has_keyword
@staticmethod
def pandas_rolling_method(df, rolling_kwargs, name, *args, groupby_kwargs=None, groupby_slice=None, **kwargs):
    groupby = df.groupby(**groupby_kwargs)
    if groupby_slice:
        groupby = groupby[groupby_slice]
    rolling = groupby.rolling(**rolling_kwargs)
    return getattr(rolling, name)(*args, **kwargs).sort_index(level=-1)