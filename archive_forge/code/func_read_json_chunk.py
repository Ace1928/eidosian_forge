from __future__ import annotations
import io
import os
from functools import partial
from itertools import zip_longest
import pandas as pd
from fsspec.core import open_files
from dask.base import compute as dask_compute
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_VERSION
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_delayed
from dask.dataframe.utils import insert_meta_param_description, make_meta
from dask.delayed import delayed
def read_json_chunk(chunk, encoding, errors, engine, column_name, path, path_dtype, kwargs, meta=None):
    s = io.StringIO(chunk.decode(encoding, errors))
    s.seek(0)
    df = engine(s, orient='records', lines=True, **kwargs)
    if meta is not None and df.empty:
        return meta
    if column_name:
        df = add_path_column(df, column_name, path, path_dtype)
    return df