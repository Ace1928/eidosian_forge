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
def check_backend_context_manager(context, backend_name):
    with context(backend_name, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert active_n_jobs == 3
        assert effective_n_jobs(3) == 3
        p = Parallel()
        assert p.n_jobs == 3
        if backend_name == 'multiprocessing':
            assert type(active_backend) is MultiprocessingBackend
            assert type(p._backend) is MultiprocessingBackend
        elif backend_name == 'loky':
            assert type(active_backend) is LokyBackend
            assert type(p._backend) is LokyBackend
        elif backend_name == 'threading':
            assert type(active_backend) is ThreadingBackend
            assert type(p._backend) is ThreadingBackend
        elif backend_name.startswith('test_'):
            assert type(active_backend) is FakeParallelBackend
            assert type(p._backend) is FakeParallelBackend