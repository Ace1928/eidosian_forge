from __future__ import annotations
import pytest
import asyncio
import os
import sys
from functools import partial
from operator import add
from distributed import Client, SchedulerPlugin, WorkerPlugin
from distributed.utils_test import cleanup  # noqa F401
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import (  # noqa F401
import dask
import dask.bag as db
from dask import compute, delayed, persist
from dask.base import compute_as_if_collection, get_scheduler
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.distributed import futures_of, wait
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.utils import get_named_args, get_scheduler_lock, tmpdir, tmpfile
from dask.utils_test import inc
@ignore_sync_scheduler_warning
@pytest.mark.parametrize('io', ['parquet-pyarrow', pytest.param('parquet-fastparquet', marks=pytest.mark.skip_with_pyarrow_strings), 'csv', pytest.param('hdf', marks=pytest.mark.flaky(reruns=5))])
@pytest.mark.parametrize('fuse', [True, False, None])
@pytest.mark.parametrize('from_futures', [True, False])
def test_blockwise_dataframe_io(c, tmpdir, io, fuse, from_futures):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.xfail("doesn't work yet")
    df = pd.DataFrame({'x': [1, 2, 3] * 5, 'y': range(15)})
    if from_futures:
        parts = [df.iloc[:5], df.iloc[5:10], df.iloc[10:15]]
        futs = c.scatter(parts)
        ddf0 = dd.from_delayed(futs, meta=parts[0])
    else:
        ddf0 = dd.from_pandas(df, npartitions=3)
    if io == 'parquet-pyarrow':
        pytest.importorskip('pyarrow')
        ddf0.to_parquet(str(tmpdir))
        ddf = dd.read_parquet(str(tmpdir))
    elif io == 'parquet-fastparquet':
        pytest.importorskip('fastparquet')
        with pytest.warns(FutureWarning):
            ddf0.to_parquet(str(tmpdir), engine='fastparquet')
            ddf = dd.read_parquet(str(tmpdir), engine='fastparquet')
    elif io == 'csv':
        ddf0.to_csv(str(tmpdir), index=False)
        ddf = dd.read_csv(os.path.join(str(tmpdir), '*'))
    elif io == 'hdf':
        pytest.importorskip('tables')
        fn = str(tmpdir.join('h5'))
        ddf0.to_hdf(fn, '/data*')
        ddf = dd.read_hdf(fn, '/data*')
    else:
        raise AssertionError('unreachable')
    df = df[['x']] + 10
    ddf = ddf[['x']] + 10
    if not dd._dask_expr_enabled():
        with dask.config.set({'optimization.fuse.active': fuse}):
            ddf.compute()
            dsk = dask.dataframe.optimize(ddf.dask, ddf.__dask_keys__())
            assert isinstance(dsk, dict) == bool(fuse)
            dd.assert_eq(ddf, df, check_index=False)