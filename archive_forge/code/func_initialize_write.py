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
def initialize_write(cls, df, fs, path, append=False, partition_on=None, ignore_divisions=False, division_info=None, schema='infer', object_encoding='utf8', index_cols=None, custom_metadata=None, **kwargs):
    if index_cols is None:
        index_cols = []
    if append and division_info is None:
        ignore_divisions = True
    fs.mkdirs(path, exist_ok=True)
    if object_encoding == 'infer' or (isinstance(object_encoding, dict) and 'infer' in object_encoding.values()):
        raise ValueError('"infer" not allowed as object encoding, because this required data in memory.')
    metadata_file_exists = False
    if append:
        try:
            pf = fastparquet.api.ParquetFile(path, open_with=fs.open)
            metadata_file_exists = fs.exists(fs.sep.join([path, '_metadata']))
        except (OSError, ValueError):
            append = False
    if append:
        from dask.dataframe._pyarrow import to_object_string
        if pf.file_scheme not in ['hive', 'empty', 'flat']:
            raise ValueError('Requested file scheme is hive, but existing file scheme is not.')
        elif set(pf.columns) != set(df.columns) - set(partition_on) or set(partition_on) != set(pf.cats):
            raise ValueError('Appended columns not the same.\nPrevious: {} | New: {}'.format(pf.columns, list(df.columns)))
        elif (pd.Series(pf.dtypes).loc[pf.columns] != to_object_string(df[pf.columns]._meta).dtypes).any():
            raise ValueError('Appended dtypes differ.\n{}'.format(set(pf.dtypes.items()) ^ set(df.dtypes.items())))
        else:
            df = df[pf.columns + partition_on]
        fmd = pf.fmd
        i_offset = fastparquet.writer.find_max_part(fmd.row_groups)
        if not ignore_divisions:
            if not set(index_cols).intersection([division_info['name']]):
                ignore_divisions = True
        if not ignore_divisions:
            minmax = fastparquet.api.sorted_partitioned_columns(pf)
            old_end = minmax[index_cols[0]]['max'][-1] if index_cols[0] in minmax else None
            divisions = division_info['divisions']
            if old_end is not None and divisions[0] <= old_end:
                raise ValueError('The divisions of the appended dataframe overlap with previously written divisions. If this is desired, set ``ignore_divisions=True`` to append anyway.\n- End of last written partition: {old_end}\n- Start of first new partition: {divisions[0]}')
    else:
        fmd = fastparquet.writer.make_metadata(df._meta, object_encoding=object_encoding, index_cols=index_cols, ignore_columns=partition_on, **kwargs)
        i_offset = 0
    if custom_metadata is not None:
        kvm = fmd.key_value_metadata or []
        kvm.extend([fastparquet.parquet_thrift.KeyValue(key=key, value=value) for key, value in custom_metadata.items()])
        fmd.key_value_metadata = kvm
    extra_write_kwargs = {'fmd': fmd}
    return (i_offset, fmd, metadata_file_exists, extra_write_kwargs)