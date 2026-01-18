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
@pytest.mark.parametrize('filters', [None, [[('b', '==', 'dog')]]])
def test_non_categorical_partitioning_pyarrow(tmpdir, filters):
    from pyarrow.dataset import partitioning as pd_partitioning
    df1 = pd.DataFrame({'a': range(100), 'b': ['cat', 'dog'] * 50})
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf1.to_parquet(path=tmpdir, partition_on=['b'], write_index=False)
    schema = pa.schema([('b', pa.string())])
    partitioning = dict(flavor='hive', schema=schema)
    ddf = dd.read_parquet(tmpdir, dataset={'partitioning': partitioning}, filters=filters)
    pdf = pd.read_parquet(tmpdir, partitioning=pd_partitioning(**partitioning), filters=filters)
    assert_eq(ddf, pdf, check_index=False)
    assert ddf['b'].dtype != 'category'