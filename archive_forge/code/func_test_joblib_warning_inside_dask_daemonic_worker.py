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
@pytest.mark.parametrize('backend', ['loky', 'multiprocessing'])
def test_joblib_warning_inside_dask_daemonic_worker(backend):
    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)

    def func_using_joblib_parallel():
        with warnings.catch_warnings(record=True) as record:
            Parallel(n_jobs=2, backend=backend)((delayed(inc)(i) for i in range(10)))
        return record
    fut = client.submit(func_using_joblib_parallel)
    record = fut.result()
    assert len(record) == 1
    warning = record[0].message
    assert isinstance(warning, UserWarning)
    assert 'distributed.worker.daemon' in str(warning)