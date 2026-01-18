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
def process_statistics(parts, statistics, filters, index, blocksize, split_row_groups, fs, aggregation_depth):
    """Process row-group column statistics in metadata
    Used in read_parquet.
    """
    if statistics and len(parts) != len(statistics):
        warnings.warn(f'Length of partition statistics ({len(statistics)}) does not match the partition count ({len(parts)}). This may indicate a bug or incorrect read_parquet usage. We must ignore the statistics and disable: filtering, divisions, and/or file aggregation.')
        statistics = []
    divisions = None
    if statistics:
        result = list(zip(*[(part, stats) for part, stats in zip(parts, statistics) if stats['num-rows'] > 0]))
        parts, statistics = result or [[], []]
        if filters:
            parts, statistics = apply_filters(parts, statistics, filters)
        if blocksize or (split_row_groups and int(split_row_groups) > 1):
            parts, statistics = aggregate_row_groups(parts, statistics, blocksize, split_row_groups, fs, aggregation_depth)
        index = [index] if isinstance(index, str) else index
        process_columns = index if index and len(index) == 1 else None
        if filters:
            process_columns = None
        if process_columns or filters:
            sorted_col_names = []
            for sorted_column_info in sorted_columns(statistics, columns=process_columns):
                if index and sorted_column_info['name'] in index:
                    divisions = sorted_column_info['divisions']
                    break
                else:
                    sorted_col_names.append(sorted_column_info['name'])
            if index is None and sorted_col_names:
                assert bool(filters)
                warnings.warn(f'Sorted columns detected: {sorted_col_names}\nUse the `index` argument to set a sorted column as your index to create a DataFrame collection with known `divisions`.', UserWarning)
    divisions = divisions or (None,) * (len(parts) + 1)
    return (parts, divisions, index)