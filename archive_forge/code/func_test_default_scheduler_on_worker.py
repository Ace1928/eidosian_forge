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
@pytest.mark.parametrize('computation', [None, 'compute_as_if_collection', 'dask.compute'])
@pytest.mark.parametrize('scheduler, use_distributed', [(None, True), ('sync', False)])
def test_default_scheduler_on_worker(c, computation, use_distributed, scheduler):
    """Should a collection use its default scheduler or the distributed
    scheduler when being computed within a task?
    """
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')

    class UpdateGraphCounter(SchedulerPlugin):

        async def start(self, scheduler):
            scheduler._update_graph_count = 0

        def update_graph(self, scheduler, *args, **kwargs):
            scheduler._update_graph_count += 1
    c.register_plugin(UpdateGraphCounter())

    def foo():
        size = 10
        df = pd.DataFrame({'x': range(size), 'y': range(size)})
        ddf = dd.from_pandas(df, npartitions=2)
        if computation is None:
            ddf.compute(scheduler=scheduler)
        elif computation == 'dask.compute':
            dask.compute(ddf, scheduler=scheduler)
        elif computation == 'compute_as_if_collection':
            compute_as_if_collection(ddf.__class__, ddf.dask, list(ddf.dask), scheduler=scheduler)
        else:
            assert False
        return True
    res = c.submit(foo)
    assert res.result() is True
    num_update_graphs = c.run_on_scheduler(lambda dask_scheduler: dask_scheduler._update_graph_count)
    assert num_update_graphs == 2 if use_distributed else 1, num_update_graphs