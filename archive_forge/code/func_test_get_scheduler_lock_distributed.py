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
@pytest.mark.parametrize('multiprocessing_method', ['spawn', 'fork', 'forkserver'])
def test_get_scheduler_lock_distributed(c, multiprocessing_method):
    da = pytest.importorskip('dask.array', reason='Requires dask.array')
    dd = pytest.importorskip('dask.dataframe', reason='Requires dask.dataframe')
    darr = da.ones((100,))
    ddf = dd.from_dask_array(darr, columns=['x'])
    dbag = db.range(100, npartitions=2)
    with dask.config.set({'distributed.worker.multiprocessing-method': multiprocessing_method}):
        for collection in (ddf, darr, dbag):
            res = get_scheduler_lock(collection, scheduler='distributed')
            assert isinstance(res, distributed.lock.Lock)