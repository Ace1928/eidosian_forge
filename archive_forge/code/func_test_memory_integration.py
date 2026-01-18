import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash
def test_memory_integration(tmpdir):
    """ Simple test of memory lazy evaluation.
    """
    accumulator = list()

    def f(arg):
        accumulator.append(1)
        return arg
    check_identity_lazy(f, accumulator, tmpdir.strpath)
    for compress in (False, True):
        for mmap_mode in ('r', None):
            memory = Memory(location=tmpdir.strpath, verbose=10, mmap_mode=mmap_mode, compress=compress)
            shutil.rmtree(tmpdir.strpath, ignore_errors=True)
            g = memory.cache(f)
            g(1)
            g.clear(warn=False)
            current_accumulator = len(accumulator)
            out = g(1)
        assert len(accumulator) == current_accumulator + 1
        assert memory.eval(f, 1) == out
        assert len(accumulator) == current_accumulator + 1
    f.__module__ = '__main__'
    memory = Memory(location=tmpdir.strpath, verbose=0)
    memory.cache(f)(1)