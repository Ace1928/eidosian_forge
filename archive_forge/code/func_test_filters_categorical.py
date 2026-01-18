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
def test_filters_categorical(tmpdir, write_engine, read_engine):
    tmpdir = str(tmpdir)
    cats = ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']
    dftest = pd.DataFrame({'dummy': [1, 1, 1, 1], 'DatePart': pd.Categorical(cats, categories=cats, ordered=True)})
    ddftest = dd.from_pandas(dftest, npartitions=4).set_index('dummy')
    ddftest.to_parquet(tmpdir, partition_on='DatePart', engine=write_engine)
    ddftest_read = dd.read_parquet(tmpdir, index='dummy', engine=read_engine, filters=[('DatePart', '<=', '2018-01-02')], calculate_divisions=True)
    assert len(ddftest_read) == 2