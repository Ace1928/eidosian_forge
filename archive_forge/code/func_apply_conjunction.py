from __future__ import annotations
import contextlib
import math
import warnings
from typing import Literal
import pandas as pd
import tlz as toolz
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
import dask
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.core import DataFrame, Scalar
from dask.dataframe.io.io import from_map
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import DataFrameIOFunction, _is_local_fs
from dask.dataframe.methods import concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import apply, import_required, natural_sort_key, parse_bytes
def apply_conjunction(parts, statistics, conjunction):
    for column, operator, value in conjunction:
        if operator not in _supported_operators:
            raise ValueError(f'"{(column, operator, value)}" is not a valid operator in predicates.')
        elif operator in ('in', 'not in') and (not isinstance(value, (list, set, tuple))):
            raise TypeError("Value of 'in' filter must be a list, set, or tuple.")
        out_parts = []
        out_statistics = []
        for part, stats in zip(parts, statistics):
            if 'filter' in stats and stats['filter']:
                continue
            try:
                c = toolz.groupby('name', stats['columns'])[column][0]
                min = c['min']
                max = c['max']
                null_count = c.get('null_count', None)
            except KeyError:
                out_parts.append(part)
                out_statistics.append(stats)
            else:
                if min is None and max is None and (not null_count) or (operator == 'is' and null_count) or (operator == 'is not' and (not pd.isna(min) or not pd.isna(max))) or (operator != 'is not' and min is None and (max is None) and null_count) or (operator in ('==', '=') and min <= value <= max) or (operator == '!=' and (null_count or min != value or max != value)) or (operator == '<' and min < value) or (operator == '<=' and min <= value) or (operator == '>' and max > value) or (operator == '>=' and max >= value) or (operator == 'in' and any((min <= item <= max for item in value))) or (operator == 'not in' and (not any((min == max == item for item in value)))):
                    out_parts.append(part)
                    out_statistics.append(stats)
        parts, statistics = (out_parts, out_statistics)
    return (parts, statistics)