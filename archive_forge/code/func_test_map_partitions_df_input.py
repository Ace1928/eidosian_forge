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
def test_map_partitions_df_input():
    """
    Check that map_partitions can handle a delayed
    partition of a dataframe input
    """
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.skip("map partitions can't deal with delayed properly")

    def f(d, a):
        assert isinstance(d, pd.DataFrame)
        assert isinstance(a, pd.DataFrame)
        return d

    def main():
        item_df = dd.from_pandas(pd.DataFrame({'a': range(10)}), npartitions=1)
        ddf = item_df.to_delayed()[0].persist()
        merged_df = dd.from_pandas(pd.DataFrame({'b': range(10)}), npartitions=1)
        merged_df = merged_df.shuffle(on='b', shuffle_method='tasks')
        merged_df.map_partitions(f, ddf, meta=merged_df, enforce_metadata=False).compute()
    with distributed.LocalCluster(scheduler_port=0, dashboard_address=':0', scheduler_kwargs={'dashboard': False}, asynchronous=False, n_workers=1, nthreads=1, processes=False) as cluster:
        with distributed.Client(cluster, asynchronous=False):
            main()