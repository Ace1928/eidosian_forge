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
@pytest.mark.parametrize('calculate_divisions', [True, False, None])
def test_ignore_metadata_file(tmpdir, engine, calculate_divisions):
    tmpdir = str(tmpdir)
    dataset_with_bad_metadata = os.path.join(tmpdir, 'data1')
    dataset_without_metadata = os.path.join(tmpdir, 'data2')
    df1 = pd.DataFrame({'a': range(100), 'b': ['dog', 'cat'] * 50})
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf1.to_parquet(path=dataset_with_bad_metadata, engine=engine, write_metadata_file=False)
    ddf1.to_parquet(path=dataset_without_metadata, engine=engine, write_metadata_file=False)
    assert '_metadata' not in os.listdir(dataset_with_bad_metadata)
    with open(os.path.join(dataset_with_bad_metadata, '_metadata'), 'w') as f:
        f.write('INVALID METADATA')
    assert '_metadata' in os.listdir(dataset_with_bad_metadata)
    assert '_metadata' not in os.listdir(dataset_without_metadata)
    ddf2a = dd.read_parquet(dataset_with_bad_metadata, engine=engine, ignore_metadata_file=True, calculate_divisions=calculate_divisions)
    ddf2b = dd.read_parquet(dataset_without_metadata, engine=engine, ignore_metadata_file=True, calculate_divisions=calculate_divisions)
    assert_eq(ddf2a, ddf2b)