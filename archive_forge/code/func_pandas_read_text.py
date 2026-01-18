from __future__ import annotations
import os
from collections.abc import Mapping
from io import BytesIO
from warnings import catch_warnings, simplefilter, warn
import numpy as np
import pandas as pd
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths
from fsspec.core import open as open_file
from fsspec.core import open_files
from fsspec.utils import infer_compression
from pandas.api.types import (
from dask.base import tokenize
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_map
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.dataframe.utils import clear_known_categories
from dask.delayed import delayed
from dask.utils import asciitable, parse_bytes
from the start of the file (or of the first file if it's a glob). Usually this
from dask.dataframe.core import _Frame
def pandas_read_text(reader, b, header, kwargs, dtypes=None, columns=None, write_header=True, enforce=False, path=None):
    """Convert a block of bytes to a Pandas DataFrame

    Parameters
    ----------
    reader : callable
        ``pd.read_csv`` or ``pd.read_table``.
    b : bytestring
        The content to be parsed with ``reader``
    header : bytestring
        An optional header to prepend to ``b``
    kwargs : dict
        A dictionary of keyword arguments to be passed to ``reader``
    dtypes : dict
        dtypes to assign to columns
    path : tuple
        A tuple containing path column name, path to file, and an ordered list of paths.

    See Also
    --------
    dask.dataframe.csv.read_pandas_from_bytes
    """
    bio = BytesIO()
    if write_header and (not b.startswith(header.rstrip())):
        bio.write(header)
    bio.write(b)
    bio.seek(0)
    df = reader(bio, **kwargs)
    if dtypes:
        coerce_dtypes(df, dtypes)
    if enforce and columns and (list(df.columns) != list(columns)):
        raise ValueError('Columns do not match', df.columns, columns)
    if path:
        colname, path, paths = path
        code = paths.index(path)
        df = df.assign(**{colname: pd.Categorical.from_codes(np.full(len(df), code), paths)})
    return df