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
def test_blockwise_different_optimization(c):
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')
    u = da.from_array(np.arange(3))
    v = da.from_array(np.array([10 + 2j, 7 - 3j, 8 + 1j]))
    cv = v.conj()
    x = u * cv
    cv, = dask.optimize(cv)
    y = u * cv
    expected = np.array([0 + 0j, 7 + 3j, 16 - 2j])
    with dask.config.set({'optimization.fuse.active': False}):
        x_value = x.compute()
        y_value = y.compute()
    np.testing.assert_equal(x_value, expected)
    np.testing.assert_equal(y_value, expected)