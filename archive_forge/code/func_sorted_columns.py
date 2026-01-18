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
def sorted_columns(statistics, columns=None):
    """Find sorted columns given row-group statistics

    This finds all columns that are sorted, along with the
    appropriate ``divisions`` for those columns. If the (optional)
    ``columns`` argument is used, the search will be restricted
    to the specified column set.

    Returns
    -------
    out: List of {'name': str, 'divisions': List[str]} dictionaries
    """
    if not statistics:
        return []
    out = []
    for i, c in enumerate(statistics[0]['columns']):
        if columns and c['name'] not in columns:
            continue
        if not all(('min' in s['columns'][i] and 'max' in s['columns'][i] for s in statistics)):
            continue
        divisions = [c['min']]
        max = c['max']
        success = c['min'] is not None
        for stats in statistics[1:]:
            c = stats['columns'][i]
            if c['min'] is None:
                success = False
                break
            if c['min'] >= max:
                divisions.append(c['min'])
                max = c['max']
            else:
                success = False
                break
        if success:
            divisions.append(max)
            assert divisions == sorted(divisions)
            out.append({'name': c['name'], 'divisions': divisions})
    return out