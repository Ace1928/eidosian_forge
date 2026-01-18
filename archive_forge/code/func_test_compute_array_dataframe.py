from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
@pytest.mark.skipif('not dd or not da')
def test_compute_array_dataframe():
    arr = np.arange(100).reshape((10, 10))
    darr = da.from_array(arr, chunks=(5, 5)) + 1
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 5, 3, 3]})
    ddf = dd.from_pandas(df, npartitions=2).a + 2
    arr_out, df_out = compute(darr, ddf)
    assert np.allclose(arr_out, arr + 1)
    dd.utils.assert_eq(df_out, df.a + 2)