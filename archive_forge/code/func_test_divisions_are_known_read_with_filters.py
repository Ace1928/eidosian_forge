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
@FASTPARQUET_MARK
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_divisions_are_known_read_with_filters(tmpdir):
    tmpdir = str(tmpdir)
    df = pd.DataFrame({'unique': [0, 0, 1, 1, 2, 2, 3, 3], 'id': ['id1', 'id2', 'id1', 'id2', 'id1', 'id2', 'id1', 'id2']}, index=[0, 0, 1, 1, 2, 2, 3, 3])
    d = dd.from_pandas(df, npartitions=2)
    d.to_parquet(tmpdir, partition_on=['id'], engine='fastparquet')
    out = dd.read_parquet(tmpdir, engine='fastparquet', filters=[('id', '==', 'id1')], calculate_divisions=True)
    assert out.known_divisions
    expected_divisions = (0, 2, 3)
    assert out.divisions == expected_divisions