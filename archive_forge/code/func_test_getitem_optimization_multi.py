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
def test_getitem_optimization_multi(tmpdir, engine):
    df = pd.DataFrame({'A': [1] * 100, 'B': [2] * 100, 'C': [3] * 100, 'D': [4] * 100})
    ddf = dd.from_pandas(df, 2)
    fn = os.path.join(str(tmpdir))
    ddf.to_parquet(fn, engine=engine)
    a = dd.read_parquet(fn, engine=engine)['B']
    b = dd.read_parquet(fn, engine=engine)[['C']]
    c = dd.read_parquet(fn, engine=engine)[['C', 'A']]
    a1, a2, a3 = dask.compute(a, b, c)
    b1, b2, b3 = dask.compute(a, b, c, optimize_graph=False)
    assert_eq(a1, b1)
    assert_eq(a2, b2)
    assert_eq(a3, b3)