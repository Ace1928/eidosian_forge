from __future__ import print_function, division, absolute_import
import os
import warnings
import pytest
from random import random
from uuid import uuid4
from time import sleep
from .. import Parallel, delayed, parallel_config
from ..parallel import ThreadingBackend, AutoBatchingMixin
from .._dask import DaskDistributedBackend
from distributed import Client, LocalCluster, get_client  # noqa: E402
from distributed.metrics import time  # noqa: E402
from distributed.utils_test import cluster, inc  # noqa: E402
def test_auto_scatter(loop_in_thread):
    np = pytest.importorskip('numpy')
    data1 = np.ones(int(10000.0), dtype=np.uint8)
    data2 = np.ones(int(10000.0), dtype=np.uint8)
    data_to_process = [data1] * 3 + [data2] * 3
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop_in_thread) as client:
            with parallel_config(backend='dask'):
                Parallel()((delayed(noop)(data, data, i, opt=data) for i, data in enumerate(data_to_process)))
            counts = count_events('receive-from-scatter', client)
            assert counts[a['address']] + counts[b['address']] == 2
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop_in_thread) as client:
            with parallel_config(backend='dask'):
                Parallel()((delayed(noop)(data1[:3], i) for i in range(5)))
            counts = count_events('receive-from-scatter', client)
            assert counts[a['address']] == 0
            assert counts[b['address']] == 0