from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@classmethod
def clamp_arrow_datetimes(cls, arrow_table: pa.Table) -> pa.Table:
    """Constrain datetimes to be valid for pandas

            Since pandas works in ns precision and arrow / parquet defaults to ms
            precision we need to clamp our datetimes to something reasonable"""
    new_columns = []
    for col in arrow_table.columns:
        if pa.types.is_timestamp(col.type) and col.type.unit in ('s', 'ms', 'us'):
            multiplier = {'s': 10000000000, 'ms': 1000000, 'us': 1000}[col.type.unit]
            original_type = col.type
            series: pd.Series = col.cast(pa.int64()).to_pandas()
            info = np.iinfo(np.dtype('int64'))
            series.clip(lower=info.min // multiplier + 1, upper=info.max // multiplier, inplace=True)
            new_array = pa.array(series, pa.int64())
            new_array = new_array.cast(original_type)
            new_columns.append(new_array)
        else:
            new_columns.append(col)
    return pa.Table.from_arrays(new_columns, names=arrow_table.column_names)