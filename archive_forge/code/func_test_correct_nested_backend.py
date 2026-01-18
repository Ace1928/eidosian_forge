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
def test_correct_nested_backend(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as client:
            with parallel_config(backend='dask'):
                result = Parallel(n_jobs=2)((delayed(outer)(nested_require=None) for _ in range(1)))
                assert isinstance(result[0][0][0], DaskDistributedBackend)
            with parallel_config(backend='dask'):
                result = Parallel(n_jobs=2)((delayed(outer)(nested_require='sharedmem') for _ in range(1)))
                assert isinstance(result[0][0][0], ThreadingBackend)