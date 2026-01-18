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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason="Can't hash metadata file at the moment")
@PYARROW_MARK
def test_custom_filename_works_with_pyarrow_when_append_is_true(tmpdir):
    fn = str(tmpdir)
    pdf = pd.DataFrame({'num1': [1, 2, 3, 4], 'num2': [7, 8, 9, 10]})
    df = dd.from_pandas(pdf, npartitions=2)
    df.to_parquet(fn, write_metadata_file=True, name_function=lambda x: f'hi-{x * 2}.parquet')
    pdf = pd.DataFrame({'num1': [33], 'num2': [44]})
    df = dd.from_pandas(pdf, npartitions=1)
    df.to_parquet(fn, name_function=lambda x: f'hi-{x * 2}.parquet', append=True, ignore_divisions=True)
    files = os.listdir(fn)
    assert '_common_metadata' in files
    assert '_metadata' in files
    assert 'hi-0.parquet' in files
    assert 'hi-2.parquet' in files
    assert 'hi-4.parquet' in files
    expected_pdf = pd.DataFrame({'num1': [1, 2, 3, 4, 33], 'num2': [7, 8, 9, 10, 44]})
    actual = dd.read_parquet(fn, index=False)
    assert_eq(actual, expected_pdf, check_index=False)