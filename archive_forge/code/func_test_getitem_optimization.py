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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="doesn't make sense")
@pytest.mark.parametrize('preserve_index', [True, False])
@pytest.mark.parametrize('index', [None, np.random.permutation(2000)])
def test_getitem_optimization(tmpdir, engine, preserve_index, index):
    tmp_path_rd = str(tmpdir.mkdir('read'))
    tmp_path_wt = str(tmpdir.mkdir('write'))
    df = pd.DataFrame({'A': [1, 2] * 1000, 'B': [3, 4] * 1000, 'C': [5, 6] * 1000}, index=index)
    df.index.name = 'my_index'
    ddf = dd.from_pandas(df, 2, sort=False)
    ddf.to_parquet(tmp_path_rd, engine=engine, write_index=preserve_index)
    ddf = dd.read_parquet(tmp_path_rd, engine=engine)['B']
    out = ddf.to_frame().to_parquet(tmp_path_wt, engine=engine, compute=False)
    dsk = optimize_dataframe_getitem(out.dask, keys=[out.key])
    subgraph_rd = hlg_layer(dsk, 'read-parquet')
    assert isinstance(subgraph_rd, DataFrameIOLayer)
    assert subgraph_rd.columns == ['B']
    assert next(iter(subgraph_rd.dsk.values()))[0].columns == ['B']
    subgraph_wt = hlg_layer(dsk, 'to-parquet')
    assert isinstance(subgraph_wt, Blockwise)
    assert_eq(ddf.compute(optimize_graph=False), ddf.compute())