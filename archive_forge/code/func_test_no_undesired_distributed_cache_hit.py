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
def test_no_undesired_distributed_cache_hit(loop):
    lists = [[] for _ in range(100)]
    np = pytest.importorskip('numpy')
    X = np.arange(int(1000000.0))

    def isolated_operation(list_, data=None):
        if data is not None:
            np.testing.assert_array_equal(data, X)
        list_.append(uuid4().hex)
        return list_
    cluster = LocalCluster(n_workers=1, threads_per_worker=2)
    client = Client(cluster)
    try:
        with parallel_config(backend='dask'):
            res = Parallel()((delayed(isolated_operation)(list_) for list_ in lists))
        assert lists == [[] for _ in range(100)]
        counts = count_events('receive-from-scatter', client)
        assert sum(counts.values()) == 0
        assert all([len(r) == 1 for r in res])
        with parallel_config(backend='dask'):
            res = Parallel()((delayed(isolated_operation)(list_, data=X) for list_ in lists))
        counts = count_events('receive-from-scatter', client)
        assert sum(counts.values()) > 0
        assert all([len(r) == 1 for r in res])
    finally:
        client.close()
        cluster.close()