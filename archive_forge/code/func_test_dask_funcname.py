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
@pytest.mark.parametrize('mixed', [True, False])
def test_dask_funcname(loop, mixed):
    from joblib._dask import Batch
    if not mixed:
        tasks = [delayed(inc)(i) for i in range(4)]
        batch_repr = 'batch_of_inc_4_calls'
    else:
        tasks = [delayed(abs)(i) if i % 2 else delayed(inc)(i) for i in range(4)]
        batch_repr = 'mixed_batch_of_inc_4_calls'
    assert repr(Batch(tasks)) == batch_repr
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_config(backend='dask'):
                _ = Parallel(batch_size=2, pre_dispatch='all')(tasks)

            def f(dask_scheduler):
                return list(dask_scheduler.transition_log)
            batch_repr = batch_repr.replace('4', '2')
            log = client.run_on_scheduler(f)
            assert all(('batch_of_inc' in tup[0] for tup in log))