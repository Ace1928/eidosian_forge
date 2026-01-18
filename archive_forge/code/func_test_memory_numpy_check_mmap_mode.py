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
@with_numpy
def test_memory_numpy_check_mmap_mode(tmpdir, monkeypatch):
    """Check that mmap_mode is respected even at the first call"""
    memory = Memory(location=tmpdir.strpath, mmap_mode='r', verbose=0)

    @memory.cache()
    def twice(a):
        return a * 2
    a = np.ones(3)
    b = twice(a)
    c = twice(a)
    assert isinstance(c, np.memmap)
    assert c.mode == 'r'
    assert isinstance(b, np.memmap)
    assert b.mode == 'r'
    del b
    del c
    gc.collect()
    corrupt_single_cache_item(memory)
    recorded_warnings = monkeypatch_cached_func_warn(twice, monkeypatch)
    d = twice(a)
    assert len(recorded_warnings) == 1
    exception_msg = 'Exception while loading results'
    assert exception_msg in recorded_warnings[0]
    assert isinstance(d, np.memmap)
    assert d.mode == 'r'