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
def test_blockwise_parquet_annotations(tmpdir, engine):
    df = pd.DataFrame({'a': np.arange(40, dtype=np.int32)})
    expect = dd.from_pandas(df, npartitions=2)
    expect.to_parquet(str(tmpdir), engine=engine)
    with dask.annotate(foo='bar'):
        ddf = dd.read_parquet(str(tmpdir), engine=engine)
    layers = ddf.__dask_graph__().layers
    assert len(layers) == 1
    layer = next(iter(layers.values()))
    assert isinstance(layer, DataFrameIOLayer)
    assert layer.annotations == {'foo': 'bar'}