import os
import sys
import time
import mmap
import weakref
import warnings
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from contextlib import nullcontext
from multiprocessing import TimeoutError
import pytest
import joblib
from joblib import parallel
from joblib import dump, load
from joblib._multiprocessing_helpers import mp
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.common import IS_PYPY, force_gc_pypy
from joblib.testing import (parametrize, raises, check_subprocess_call,
from queue import Queue
from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend
from joblib.parallel import Parallel, delayed
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import register_parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count
from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND
from joblib import Parallel, delayed
import sys
from joblib import Parallel, delayed
import sys
import faulthandler
from joblib import Parallel, delayed
from functools import partial
import sys
from joblib import Parallel, delayed, hash
import multiprocessing as mp
@with_numpy
@skipif(distributed is None, reason='This test requires dask')
@parametrize('context', [parallel_config, parallel_backend])
def test_nested_parallelism_with_dask(context):
    client = distributed.Client(n_workers=2, threads_per_worker=2)
    data = np.ones(int(10000000.0), dtype=np.uint8)
    for i in range(2):
        with context('dask'):
            backend_types_and_levels = _recursive_backend_info(data=data)
        assert len(backend_types_and_levels) == 4
        assert all((name == 'DaskDistributedBackend' for name, _ in backend_types_and_levels))
    with context('dask'):
        backend_types_and_levels = _recursive_backend_info()
    assert len(backend_types_and_levels) == 4
    assert all((name == 'DaskDistributedBackend' for name, _ in backend_types_and_levels))