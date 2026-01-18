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
@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call(backend, n_jobs):
    with raises(RuntimeError, match='This Parallel instance is already running'):
        parallel = Parallel(n_jobs, backend=backend, return_as='generator')
        g = parallel((delayed(sleep)(1) for _ in range(10)))
        t_start = time.time()
        gen2 = parallel((delayed(id)(i) for i in range(100)))
    assert time.time() - t_start < 2, 'The error should be raised immediatly when submitting a new task but it took more than 2s.'
    del g
    force_gc_pypy()