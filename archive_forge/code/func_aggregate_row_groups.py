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
def aggregate_row_groups(parts, stats, blocksize, split_row_groups, fs, aggregation_depth):
    if not stats or not stats[0].get('file_path_0', None):
        return (parts, stats)
    parts_agg = []
    stats_agg = []
    use_row_group_criteria = split_row_groups and int(split_row_groups) > 1
    use_blocksize_criteria = bool(blocksize)
    if use_blocksize_criteria:
        blocksize = parse_bytes(blocksize)
    next_part, next_stat = ([parts[0].copy()], stats[0].copy())
    for i in range(1, len(parts)):
        stat, part = (stats[i], parts[i])
        same_path = stat['file_path_0'] == next_stat['file_path_0']
        multi_path_allowed = False
        if aggregation_depth:
            multi_path_allowed = len(part['piece']) == 1
            if not (same_path or multi_path_allowed):
                rgs = set(list(part['piece'][1]) + list(next_part[-1]['piece'][1]))
                multi_path_allowed = rgs == {None} or None not in rgs
            if not same_path and multi_path_allowed:
                if aggregation_depth is True:
                    multi_path_allowed = True
                elif isinstance(aggregation_depth, int):
                    root = stat['file_path_0'].split(fs.sep)[:-aggregation_depth]
                    next_root = next_stat['file_path_0'].split(fs.sep)[:-aggregation_depth]
                    multi_path_allowed = root == next_root
                else:
                    raise ValueError(f'{aggregation_depth} not supported for `aggregation_depth`')

        def _check_row_group_criteria(stat, next_stat):
            if use_row_group_criteria:
                return next_stat['num-row-groups'] + stat['num-row-groups'] <= int(split_row_groups)
            else:
                return False

        def _check_blocksize_criteria(stat, next_stat):
            if use_blocksize_criteria:
                return next_stat['total_byte_size'] + stat['total_byte_size'] <= blocksize
            else:
                return False
        stat['num-row-groups'] = stat.get('num-row-groups', 1)
        next_stat['num-row-groups'] = next_stat.get('num-row-groups', 1)
        if (same_path or multi_path_allowed) and (_check_row_group_criteria(stat, next_stat) or _check_blocksize_criteria(stat, next_stat)):
            next_piece = next_part[-1]['piece']
            this_piece = part['piece']
            if same_path and len(next_piece) > 1 and (next_piece[1] != [None]) and (this_piece[1] != [None]):
                next_piece[1].extend(this_piece[1])
            else:
                next_part.append(part)
            next_stat['total_byte_size'] += stat['total_byte_size']
            next_stat['num-rows'] += stat['num-rows']
            next_stat['num-row-groups'] += stat['num-row-groups']
            for col, col_add in zip(next_stat['columns'], stat['columns']):
                if col['name'] != col_add['name']:
                    raise ValueError('Columns are different!!')
                if 'min' in col:
                    col['min'] = min(col['min'], col_add['min'])
                if 'max' in col:
                    col['max'] = max(col['max'], col_add['max'])
        else:
            parts_agg.append(next_part)
            stats_agg.append(next_stat)
            next_part, next_stat = ([part.copy()], stat.copy())
    parts_agg.append(next_part)
    stats_agg.append(next_stat)
    return (parts_agg, stats_agg)