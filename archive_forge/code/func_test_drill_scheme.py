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
def test_drill_scheme(tmpdir):
    fn = str(tmpdir)
    N = 5
    df1 = pd.DataFrame({c: np.random.random(N) for i, c in enumerate(['a', 'b', 'c'])})
    df2 = pd.DataFrame({c: np.random.random(N) for i, c in enumerate(['a', 'b', 'c'])})
    files = []
    for d in ['test_data1', 'test_data2']:
        dn = os.path.join(fn, d)
        if not os.path.exists(dn):
            os.mkdir(dn)
        files.append(os.path.join(dn, 'data1.parq'))
    fastparquet.write(files[0], df1)
    fastparquet.write(files[1], df2)
    df = dd.read_parquet(files, engine='fastparquet')
    assert 'dir0' in df.columns
    out = df.compute()
    assert 'dir0' in out
    assert (np.unique(out.dir0) == ['test_data1', 'test_data2']).all()