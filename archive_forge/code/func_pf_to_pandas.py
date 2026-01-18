from __future__ import annotations
import copy
import pickle
import threading
import warnings
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import numpy as np
import pandas as pd
import tlz as toolz
from packaging.version import parse as parse_version
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_201
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _is_local_fs, _meta_from_dtypes, _open_input_files
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.delayed import Delayed
from dask.utils import natural_sort_key
@classmethod
def pf_to_pandas(cls, pf, fs=None, columns=None, categories=None, index=None, open_file_options=None, **kwargs):
    if columns is not None:
        columns = columns[:]
    else:
        columns = pf.columns + list(pf.cats)
    if index:
        columns += [i for i in index if i not in columns]
    rgs = pf.row_groups
    size = sum((rg.num_rows for rg in rgs))
    df, views = pf.pre_allocate(size, columns, categories, index)
    if parse_version(fastparquet.__version__) <= parse_version('2023.02.0') and PANDAS_GE_201 and df.columns.empty:
        df.columns = pd.Index([], dtype=object)
    start = 0
    fn_rg_map = defaultdict(list)
    for rg in rgs:
        fn = pf.row_group_filename(rg)
        fn_rg_map[fn].append(rg)
    precache_options, open_file_options = _process_open_file_options(open_file_options, **{'allow_precache': False, 'default_cache': 'readahead'} if _is_local_fs(fs) else {'metadata': pf, 'columns': list(set(columns).intersection(pf.columns)), 'row_groups': [rgs for rgs in fn_rg_map.values()], 'default_engine': 'fastparquet', 'default_cache': 'readahead'})
    with ExitStack() as stack:
        for fn, infile in zip(fn_rg_map.keys(), _open_input_files(list(fn_rg_map.keys()), fs=fs, context_stack=stack, precache_options=precache_options, **open_file_options)):
            for rg in fn_rg_map[fn]:
                thislen = rg.num_rows
                parts = {name: v if name.endswith('-catdef') else v[start:start + thislen] for name, v in views.items()}
                pf.read_row_group_file(rg, columns, categories, index, assign=parts, partition_meta=pf.partition_meta, infile=infile, **kwargs)
                start += thislen
    return df