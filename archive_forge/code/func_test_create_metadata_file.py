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
@PYARROW_MARK
@pytest.mark.parametrize('partition_on', [None, 'a'])
def test_create_metadata_file(tmpdir, partition_on):
    tmpdir = str(tmpdir)
    df1 = pd.DataFrame({'b': range(100), 'a': ['A', 'B', 'C', 'D'] * 25})
    df1.index.name = 'myindex'
    ddf1 = dd.from_pandas(df1, npartitions=10)
    ddf1.to_parquet(tmpdir, write_metadata_file=False, partition_on=partition_on, engine='pyarrow')
    if partition_on:
        fns = glob.glob(os.path.join(tmpdir, partition_on + '=*/*.parquet'))
    else:
        fns = glob.glob(os.path.join(tmpdir, '*.parquet'))
    dd.io.parquet.create_metadata_file(fns, engine='pyarrow', split_every=3)
    ddf2 = dd.read_parquet(tmpdir, calculate_divisions=True, split_row_groups=False, engine='pyarrow', index='myindex')
    if partition_on:
        ddf1 = df1.sort_values('b')
        ddf2 = ddf2.compute().sort_values('b')
        ddf2.a = ddf2.a.astype('object')
    assert_eq(ddf1, ddf2)
    fmd = dd.io.parquet.create_metadata_file(fns, engine='pyarrow', split_every=3, out_dir=False)
    fmd_file = pq.ParquetFile(os.path.join(tmpdir, '_metadata')).metadata
    assert fmd.num_rows == fmd_file.num_rows
    assert fmd.num_columns == fmd_file.num_columns
    assert fmd.num_row_groups == fmd_file.num_row_groups