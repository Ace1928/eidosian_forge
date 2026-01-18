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
def overlap_chunk(func, before, after, *args, **kwargs):
    dfs = [df for df in args if isinstance(df, CombinedOutput)]
    combined, prev_part_length, next_part_length = dfs[0]
    args = [arg[0] if isinstance(arg, CombinedOutput) else arg for arg in args]
    out = func(*args, **kwargs)
    if prev_part_length is None:
        before = None
    if isinstance(before, datetime.timedelta):
        before = prev_part_length
    expansion = None
    if combined.shape[0] != 0:
        expansion = out.shape[0] // combined.shape[0]
    if before and expansion:
        before *= expansion
    if next_part_length is None:
        return out.iloc[before:]
    if isinstance(after, datetime.timedelta):
        after = next_part_length
    if after and expansion:
        after *= expansion
    return out.iloc[before:-after]